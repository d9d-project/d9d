---
DEP: 0013
Title: Parameter Runtime
Author: Manushin Maxim @maloimushiy
Type: Feature
---

# Parameter Runtime

## Abstract

d9d runs data parallelism with module hooks: PyTorch FSDP2 for sharded parameters and the `GradientSynchronizer` for replicated ones. The hooks decide when to gather weights and reduce gradients. They cannot see the pipeline program, which knows the order of forward and backward work.

This DEP lets the pipeline program drive data parallelism. A model exposes execution boundaries, so the program gathers weights before each part and reduces its complete gradients after backward. One `ParameterRuntime` runs DDP, FSDP and HSDP from the placements of the parameters, and the `parallelize_*` functions only place parameters.

## Motivation

*   **Gathers repeat per microbatch**: FSDP2 gathers a module in its pre-forward hook, so GPipe gathers each layer again for every microbatch. Keeping weights after forward avoids these gathers but keeps the whole stage gathered. A window can instead gather one region for all microbatches and release it before the next region.
*   **Split backward separates input and weight gradients**: FSDP2 starts its post-backward work when autograd computes the input gradients. Zero Bubble schedules compute the weight gradients later, so the program must control their reduction and the weight lifetime.
*   **Small collectives between nodes**: FSDP2 issues one collective per `fully_shard` unit. The d9d models make four units per layer, two of them norms. In a three-node profile, these small collectives used NCCL's low-latency protocol and lost inter-node bandwidth.
*   **Two mechanisms for one concept**: DDP swaps module classes in forward hooks (`ToLocalParallel`) and reduces with its own buckets. FSDP and HSDP run through FSDP2 with forced sum reduction. Each has its own stream ordering and its own memory behavior.
*   **Static regions split the forward definition**: A region must wrap every computation between boundaries in a module. A resumable forward can express these computations in one function.

## Design Proposal

### Resumable forwards

This DEP introduces `ModuleSupportsWindows`, `WindowSteps`, `WindowBoundary`, `mark_window()` and `run_steps()` in `d9d.pipelining.api`:

```python
@typing.runtime_checkable
class ModuleSupportsWindows(typing.Protocol[TInput, TShared, TOutput]):
    """Protocol for modules with a forward pass that can suspend at execution boundaries."""

    async def forward_steps(self, steps: WindowSteps, inputs: TInput, shared: TShared) -> TOutput:
        """Computes the model, yielding execution boundaries through ``steps``."""


@dataclasses.dataclass(frozen=True, slots=True)
class WindowBoundary(typing.Generic[TState]):
    """A request to finish the current region and optionally enter another module's parameter group."""

    module: nn.Module | None
    state: TState
    prefetch: nn.Module | None = None

    def __await__(self) -> Generator[WindowBoundary[TState], TState, TState]:
        """Yields the request and returns the state supplied by the executor."""


class WindowSteps:
    """Execution boundaries belonging to one forward invocation."""

    def __init__(self, *, eager: bool = False):
        """Selects ordinary nested module calls for eager execution."""

    async def enter(self, module: nn.Module, state: TState, *, prefetch: nn.Module | None = None) -> TState:
        """Finishes the previous region and requests the module's parameter group before continuing."""

    async def finish(self, state: TState) -> TState:
        """Finishes the open region and returns the state supplied by the executor."""

    async def call(self, module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        """Runs an ordinary module in one region, or composes a resumable module's boundaries."""


def mark_window(module: TModule) -> TModule:
    """Marks a parameter group, returning the same module without wrappers or execution hooks."""


def run_steps(module: ModuleSupportsWindows[TInput, TShared, TOutput], inputs: TInput, shared: TShared) -> TOutput:
    """Runs a resumable forward eagerly, without graph cuts or parameter management."""
```

Each forward invocation receives its own `WindowSteps`. It contains no global runtime and installs no hooks. Calling `forward_steps()` creates a native coroutine. An executor advances it with `send()` until it yields a `WindowBoundary` or returns its result. Only these boundary events are supported; there is no `asyncio` event loop.

`enter(module, state)` yields before any computation that uses the requested parameters. The executor can therefore gather the first group's weights without learning an execution order beforehand. An entry also finishes the preceding open region. `finish(state)` closes a region without requesting another group; its event has `module=None`. When the executor's outer coroutine returns, it closes any remaining open region with the returned state. Returning after `finish()` does not close it again.

`call()` supplies its arguments as entry state and its output as finish state. Both states come back through the executor before the function continues. A window executor can replace tensors at these boundaries and record their producers. The final result must route its gradients to those producers, even when the last call already finished its region.

Previously returned boundary tensors keep their producers. A caller can retain such a tensor for a later `call()` without including it in intermediate calls' outputs. Every consumer contributes its gradient before the producer's backward runs.

The contracts follow from suspension and graph cuts:

*   **One owner of execution order**: `forward_steps()` specifies the actual order. Marked modules describe membership, without repeating that order.
*   **Complete boundary state**: Manual entry and finish expose every new differentiable value that later regions need. This includes residuals and collected hidden states. The caller reassigns the returned state. References outside that state, including closure and module fields, are not rewritten automatically.
*   **Module calls contain their computation**: The simple API keeps differentiable operations inside `call()`. Other tensor arithmetic belongs in an explicitly entered region.
*   **Explicit mixing**: `call()` rejects an open manual region, because its arguments might omit live state. Finish that region explicitly before calling another module. `finish()` without an open region is also an error.
*   **Contexts stop at boundaries**: User contexts such as `no_grad()` and autocast must not span a suspension. The executor raises an error if grad mode, inference mode or autocast changes across one. A window executor enters each microbatch's compute context around each resume and closes its coroutine in that context after failure or cancellation.
*   **Nested models**: In window execution, `call()` composes a resumable module and finishes any region it leaves open. Ordinary modules form one region. In eager execution, nested modules use `__call__`, preserving their forward hooks. A directly awaited helper leaves its final region open until an explicit boundary or the outer return.

`run_steps()` resumes every boundary immediately with its original state. It preserves the ordinary autograd graph and propagates exceptions. It closes the coroutine on exit and rejects any other yielded event. A synchronous `forward()` delegates to this helper, so the model defines its computation once.

### Parameter groups and execution traces

`mark_window(module)` records group membership on the module. It changes neither execution nor parameter names. `bind()` discovers these roots in the stage tree. A group's own modules exclude marked descendants. The stage root covers the remaining parameters. A model without `forward_steps()` has one group containing the whole stage.

A tied parameter has one handle and belongs to every group that uses it. Registration order determines bucket membership and identifiers, but does not determine execution order. The coroutine requests a group before reading its weights.

A region is one execution of a group. For the trace `A, B, A`, group A has two separate region graphs. Its gathered gradients are prepared before backward of the last occurrence and reduced after backward of the first occurrence.

The first forward gathers on demand and records its trace. Later forwards prefetch the next group from this trace. Manual `enter(..., prefetch=next_module)` supplies an explicit next group; the next request must match it. Every microbatch in a window and participating rank must use the same requests and hints.

The group sequence remains fixed while the runtime is bound. Local mismatches are rejected, and each forward must cover every parameter through at least one executed group. This contract permits token routing inside a group, but excludes rank-dependent group skipping. Binding again clears the learned trace.

Boundary tensors retain one producer. Duplicate references in state use the same detached leaf. Consumers add their gradients before the producer runs. The executor adds an output graph to route the loss gradient to region producers, including results of already finished calls.

### Axis roles

The runtime must know, for each mesh dimension of a parameter, whether data parallelism gathers it, sums its gradients or leaves it to the layer. Guessing from the placement alone is wrong: a `Shard(0)` of tensor parallelism must not be gathered.

This DEP introduces `DATA_SHARD_DIMS` in `d9d.dist_context`: the mesh dimensions along which data parallelism shards parameters, `DenseDim.DP_CP_SHARD`. The runtime takes them as an explicit input:

| Placement | Mesh dimension | Weight | Gradient |
|:----------|:---------------|:-------|:---------|
| `Shard(0)` | In `DATA_SHARD_DIMS` | Gathered before the region | Reduce-scatter |
| `Replicate` | Any | Present | All-reduce |
| `Shard` | Not in `DATA_SHARD_DIMS` | Left to the layer | Left to the layer |

DDP, FSDP and HSDP are then one implementation. Expert parallelism shards along `ExpertDim.EP_SHARD`, so the experts stay with the layer. A custom strategy places parameters with `distribute_tensor()` along its own dimensions, as before. `parallelize_fsdp()` and `parallelize_hsdp()` reject a shard dimension outside `DATA_SHARD_DIMS`, or one that the mesh lacks.

### Buckets

A collective per parameter is too small between nodes, so the runtime groups parameters into buckets. One bucket uses one all-gather and one reduce-scatter. A gathered bucket holds the parameters that the same groups use, so its weights live and die together. Buckets without a gather fill up to `bucket_size_mib`. Shards are padded to equal size, so every collective has the even form.

The weights of a bucket are views of one flat buffer. The runtime frees it with `untyped_storage().resize_(0)`, so the views that autograd saved stay valid. A tied parameter has one entry in one bucket, and autograd adds up the gradients of all its uses.

### Memory

Gathered weights, gradient buffers and the temporaries of collectives live in a memory budget. A ledger counts a released buffer as busy until the GPU finishes reading it, so the host waits only when the budget runs out. The gradient shards, which live for the whole step, and the `compute_dtype` copies of parameters without a gather are not counted.

The requested budget is a target with a floor. The floor covers peak group weights and live gradients, times four for prefetch and collectives. Observing a trace updates gradient lifetimes for repeated and tied uses. Already retained weights remain accounted for when the floor grows. The factor is a stopgap until the runtime measures the peak of a region. A smaller request is raised to the floor. The default request is the floor, so no weights stay gathered between regions, as in FSDP. Room above the floor keeps buckets gathered for the whole step: frozen ones first, then groups in reverse registration order.

### The runtime

This DEP introduces `ParameterRuntime` in `d9d.parallel.runtime`:

```python
class ParameterRuntime:
    """Runtime of data parallelism that the pipeline program drives."""

    def __init__(
        self,
        gather_dims: Collection[str],
        bucket_size_mib: int,
        compute_dtype: torch.dtype | None = None,
        memory_budget_bytes: int = 0,
    ):
        """Constructs the ``ParameterRuntime`` object.

        Args:
            gather_dims: The mesh dimensions along which data parallelism shards parameters.
            bucket_size_mib: The size limit of buckets without a gather, in MiB.
            compute_dtype: The dtype of the unsharded weights, or ``None`` to keep the dtype of the storage.
            memory_budget_bytes: The target budget of gathered weights, gradients and collective temporaries.
                Values below the required minimum are raised, so the default keeps no weights between regions.
        """

    def register_stage(self, stage_index: int, module: nn.Module): ...
    def bind(self, training: bool = True): ...
    def unbind(self): ...
    def group_index(self, stage_index: int, module: nn.Module | None) -> int: ...
    def group_parameters(self, stage_index: int, group: int | None) -> list[nn.Parameter]: ...
    def expected_group(self, stage_index: int, position: int) -> int | None: ...
    def record_order(self, stage_index: int, order: tuple[int, ...]): ...

    @property
    def compute_dtype(self) -> torch.dtype | None: ...

    @contextmanager
    def compute_parameters(self, stage_index: int) -> Generator[None, None, None]:
        """Puts the unsharded parameters of a stage into its modules without their values."""

    @contextmanager
    def group_weights(
        self, stage_index: int, group_index: int, prepare: Sequence[BucketId] = (), next_group: int | None = None
    ) -> Generator[None, None, None]:
        """Puts the unsharded weights of a region into its modules for the duration of the block."""

    def reduction_plan(self, stage_index: int, order: Sequence[int | None], final: bool) -> ReductionPlan: ...
    def reduce(self, plan: ReductionPlan, bucket_id: BucketId): ...
    def wait(self): ...
    def scale_grads(self, factor: torch.Tensor): ...
    def zero_grad(self): ...
    def invalidate_weights(self): ...
```

Every rank enters `group_weights()` in the same order, because it issues collectives. `PipelineProgramCache` records only which weight-writing actions are final. Before backward, the executor builds a `ReductionPlan` from the completed trace. Only the final pass also sums gradient shards over replicas. `wait()` checks that every trainable bucket had a final reduction.

### Windows

In a distributed job without pipeline parallelism, consecutive forward computes, and consecutive full backward computes, become window actions. A window gathers each region once for all its microbatches, then releases its weights before the next region. A backward window also reduces the complete gradients once, with one autograd pass per microbatch. Forward and backward windows gather separately unless the memory budget keeps the weights between them.

A window creates the compute context of each microbatch from the step handler once and enters it once per region. A compute context can therefore be entered several times, and each entry continues where the previous one stopped. The forks of `d9d.rng` do this, so a window draws the same numbers as a stage that runs in one go.

### Compute dtype

`compute_dtype` is the dtype of every unsharded weight. A gathered weight arrives in it. A parameter without a gather gets a copy, which the first region of each step refreshes. The gradients keep the gradient dtype of the parameter. `bind()` also converts the non-persistent floating-point buffers, such as rotary tables, to it, and persistent buffers keep their dtype. A stage describes its transfers inside `compute_parameters()`, so the receive buffers get the dtype that the unsharded parameters compute with.

`BucketId` and `ReductionPlan` are frozen dataclasses in `d9d.parallel.runtime`:

```python
@dataclasses.dataclass(frozen=True, slots=True)
class BucketId:
    """A bucket of one pipeline stage."""

    stage: int
    index: int


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class ReductionPlan:
    """The buckets that one backward pass reduces after each of its regions."""

    final: bool
    after_region: Mapping[int, tuple[BucketId, ...]]
    before_region: Mapping[int, tuple[BucketId, ...]]
```

### Lifecycle

1.  `register_stage()` adds each stage module. It is allowed only while the runtime is unbound, once per stage index.
2.  `bind()` reads group membership, plans buckets, allocates their buffers and converts module buffers. If it fails, the runtime is unbound again. Training binds in the gradient manager, and inference binds with `bind(training=False)` around its loop.
3.  A step enters `group_weights()` for every region on every rank and reduces each bucket after its planned region. It then calls `wait()`, `scale_grads()`, the optimizer step and `zero_grad()`, in that order.
4.  `unbind()` releases the buffers. A runtime can be bound again, e.g. after an offload.

The step methods raise `ValueError` on an unbound runtime. `invalidate_weights()`, `zero_grad()` and `unbind()` also work on an unbound runtime, because a checkpoint load or a teardown can come first.

`wait()` finishes the gradients once per step. A second call raises `ValueError`, also after a hook raised, so no hook runs twice. `zero_grad()` starts the next step.

The `final` flag of a plan marks the last backward pass of a stage in the step. Only that pass sums the gradient shards over the replicas, and `wait()` checks that every bucket had one.

### Gradients and hooks

The gradient of a sharded parameter views the gradient shard of its bucket, because `scale_grads()` and `zero_grad()` act on the shard. PyTorch lets a post-accumulate-grad hook assign a new gradient, so the runtime copies such a value back into the shard. A gradient set to `None` is an error, and `wait()` rejects a gradient that other code replaced.

A parameter that no rank used in the step gets `.grad = None`, so the optimizer skips it, as without the runtime. Each rank marks a parameter when autograd writes its gradient, and the final reduction combines the marks over every mesh dimension of the parameter. A parameter that its layer shards, such as an expert, therefore keeps a zero gradient on a rank whose part got no tokens. Reading these marks in `wait()` requires one host synchronization per step. Budget pressure can also make the memory ledger wait for buffers.

Post-accumulate-grad hooks run once per step on the sharded parameter of a used parameter, in `wait()`, as FSDP2 runs them after the reduction. Tensor hooks run on the local gradient of each backward pass. The runtime reads the hooks at each call, so a hook that removes itself, or a hook added later, takes effect at once.

### The loop

The loops build the runtime at their edges and pass it to `ModelStageFactory`, `GradientManager` and `TrackedModules`. `ParameterRuntimeConfig` configures it:

```python
class ParameterRuntimeConfig(BaseModel):
    """Configuration for the ``ParameterRuntime``, which gathers weights and reduces gradients.

    Attributes:
        compute_dtype: The dtype of the unsharded weights. If ``None``, they keep the dtype of the storage.
        memory_budget_mib: The target budget of gathered weights, gradients and collective temporaries, in MiB.
            Values below the required minimum are raised, so the default 0 keeps no weights between regions.
        bucket_size_mib: The size limit of a bucket of replicated parameters, in MiB.
    """
```

`TrainerConfig` and `InferenceConfig` get the section `parameter_runtime`. `ModelStageFactory` rejects a provider whose `parallelize_*()` calls split a tied parameter into two. `TrackedModules` takes the runtime and drops its unsharded weights before a checkpoint load.

### Verification

The implementation must keep the following measurable, with the commands and the configuration in the PR:

*   **Correctness**: Gradients and weights of DDP, FSDP, HSDP and PP with HSDP match FSDP2 and the reference within float tolerance. A checkpoint saved with FSDP2 resumes on the runtime and matches a continuous FSDP2 reference.
*   **Performance**: Steady-state step time with its spread, peak allocated memory, and the count of gathers and reductions per step, against the base commit. Include the usage reductions and the synchronization before the optimizer step. The report names both commits, the topology, the dtype, the sequence length, the global batch and the microbatch count.

## Usage

A resumable model keeps its computation in one function:

```python
from d9d.pipelining.api import WindowSteps, mark_window, run_steps


class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.first = mark_window(nn.Sequential(nn.Linear(width, width), nn.Tanh()))
        self.second = mark_window(nn.Linear(width, width))

    async def forward_steps(self, steps: WindowSteps, inputs: torch.Tensor, shared: None) -> torch.Tensor:
        hidden = await steps.call(self.first, inputs)
        return await steps.call(self.second, hidden)

    def forward(self, inputs: torch.Tensor, shared: None) -> torch.Tensor:
        return run_steps(self, inputs, shared)
```

The manual form can carry a residual between regions:

```python
async def forward_steps(self, steps: WindowSteps, inputs: torch.Tensor, shared: None) -> torch.Tensor:
    inputs = await steps.enter(self.first, inputs)
    hidden = self.first(inputs)
    hidden, skip = await steps.enter(self.second, (hidden, inputs))
    return self.second(hidden) + skip
```

The provider places parameters as before:

```python
def parallelize_model_stage(self, context: ParallelizeModelStageContext[MyModel]):
    mesh = context.dist_context.mesh_for(DENSE_DOMAIN)
    for layer in context.model.layers.values():
        parallelize_hsdp(
            layer,
            mesh=mesh[DenseDim.DP_REPLICATE, DenseDim.DP_CP_SHARD, DenseDim.CP_REPLICATE],
            shard_dim=DenseDim.DP_CP_SHARD,
        )
```

The configuration selects the precision and the budget:

```yaml
parameter_runtime:
  compute_dtype: bfloat16
  memory_budget_mib: 16384
```

## Backward Compatibility

*   **Hidden states aggregators**: `BaseHiddenStatesAggregator` does not retain the collected values. A custom aggregator implements `collect()` and `pack()` instead of `add_hidden_states()` and `pack_with_snapshot()`.
*   **Custom loops**: A custom loop creates a `ParameterRuntime`, passes it to `build_schedule()` and binds it. It calls `wait()` after each step.
*   **Configs**: `gradient_manager.bucket_size_mb` moves to `parameter_runtime.bucket_size_mib`. The old key is rejected.
*   **Parallelizers**: `parallelize_fsdp()` and `parallelize_hsdp()` take no `fully_shard()` options. The memory budget replaces `reshard_after_forward`, and `compute_dtype` with `grad_dtype` replaces the mixed precision policy.
*   **Hooks**: A post-accumulate-grad hook sees the summed gradient once per step, not the local gradient of each microbatch. For a resumable module, the window executor calls `forward_steps()` directly, so that module's forward hooks do not run. Hooks on ordinary module calls still run, and eager execution preserves nested module hooks.
*   **Sharding**: The runtime gathers only `Shard(0)` along one data shard dimension. A parameter sharded along a data shard dimension and another dimension is rejected.
*   **Compute contexts**: A window enters the compute context of a microbatch once per region. A custom `compute_context()` must return a context that can be entered again, so a generator from `@contextlib.contextmanager` does not work there.
*   **Forks of the global generators**: `fork_rng()` and `fork_rng_child()` from DEP-0011 return contexts that can be entered again and continue their numbers.
*   **Removed modules**: `d9d.parallel.grad_sync` and `d9d.parallel.style.ToLocalParallel` are removed. `TrackedModules` moves to `d9d.loop.component.tracked_modules`, and `d9d.loop.component` still exports it.

Models without resumable forwards keep working as one group. Training checkpoints keep their format, because every rank still saves the local shards of its `DTensor` parameters.

## Alternatives Considered

### Observe an ordinary forward

FSDP2 observes module calls through hooks and can use their order for prefetch. Observation alone cannot suspend a caller between regions to interleave microbatches. Native coroutines provide explicit suspension without threads, hooks or rewriting the forward function.

### Steer FSDP2 from the program

FSDP2 offers `unshard()`, `set_reshard_after_forward()` and `set_requires_gradient_sync()`. The pipeline stages of `torch.distributed.pipelining` set these flags per microbatch. The program would still need the knowledge of the runtime, DDP would stay a second mechanism, and the collectives would stay per `fully_shard` unit.

### Hooks with a counter

The `GradientSynchronizer` counts backward passes to find the last one, and DeepSpeed takes the boundary from a configured count of micro steps. The program knows the last action of a stage exactly, also with split backward, so the plan carries a `final` flag instead.

### A collective per parameter

The first prototype of this runtime issued one collective per parameter. On three nodes with eight GPUs each, NCCL selected its low-latency protocol for these messages of a few MiB. Uneven shards also turned into many small collectives. One collective per bucket took the step from 0.84 s to 0.67 s. Megatron-FSDP and FSDP2 also group parameters for one collective.

### Swap `.data` for the gathered weights

Replacing `param.data` breaks the views that autograd saved, e.g. `weight.t()` in `F.linear`. Resizing one `Storage` keeps every view valid, as FSDP2 does.

### Infer the gather dimension

A runtime could gather along any `Shard(0)` dimension. A tensor-parallel `Shard(0)` would then be gathered without an error. The explicit `DATA_SHARD_DIMS` leaves every other `Shard` to its layer.

### One autograd pass per window

A window could run one backward pass over the seeds of all its microbatches. That pass would run outside the compute contexts of the microbatches, so recomputation and custom operations would draw other numbers than without windows.

### A new fork per region

A window could open a fresh compute context for each region. Each opening would seed the generators from the same key. Every region of a microbatch would draw the same numbers, e.g. the same dropout mask.

### A budget from the GPU memory

A budget of a fraction of the GPU memory keeps more weights gathered. At `bind()`, the runtime cannot see the peaks of the activations and the allocator, so such a default can run out of memory. The floor gives the memory of FSDP, and a larger budget is a choice of the user.

## Prerequisites

Separate PRs cover split backward (#147), the forward cache (#146), optimizer resume (#144), and tied parameters and buffers (#145). Other prerequisites are the stage computations package (#148), stateless aggregators (#149) and reusable RNG forks. Checkpoint save and load outside inference mode is a separate fix (#160).

## Deferred Work

*   **Dynamic group order**: Divergent requests between microbatches or ranks need a scheduling contract. Token routing inside a fixed sequence of groups does not require this extension.
*   **Checkpointing a resumable region**: A consumed coroutine cannot replay an earlier region. Checkpointing stays inside synchronous region computations.
*   **Tensor parallelism**: TP dimensions already stay with the layer. Partial gradients of replicated TP parameters, e.g. norms with sequence parallelism, need a reduction rule.
*   **Windows with pipeline parallelism**: A window delays the output of a stage that another rank waits for. It needs a schedule that plans for this delay.
*   **Tied parameters across stages**: A parameter that two stages on different ranks share needs a gradient exchange.
*   **Other shard dimensions**: The runtime gathers only `Shard(0)`.
*   **CPU offload of parameters**: FSDP2 offered `CPUOffloadPolicy`, and the runtime has none. Sleep and wake still move the whole state to host memory between steps.
