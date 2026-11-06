---
DEP: 0012
Title: Parameter Runtime
Author: Manushin Maxim @maloimushiy
Type: Feature
---

# Parameter Runtime

## Abstract

d9d runs data parallelism with module hooks: PyTorch FSDP2 for sharded parameters and the `GradientSynchronizer` for replicated ones. The hooks decide when to gather weights and reduce gradients. They cannot see the pipeline program, which knows the order of forward and backward work.

This DEP lets the pipeline program drive data parallelism. A model describes each pipeline stage as a static chain of segments. Around each segment, the program gathers its weights. After its backward pass, the program reduces the gradients that are complete. One `ParameterRuntime` runs DDP, FSDP and HSDP from the placements of the parameters, and the `parallelize_*` functions only place parameters.

## Motivation

*   **Gathers repeat per microbatch**: FSDP2 gathers a module in its pre-forward hook, so GPipe gathers each layer again for every microbatch. Keeping weights after forward avoids these gathers but keeps the whole stage gathered. A window can instead gather one segment for all microbatches and release it before the next segment.
*   **Split backward separates input and weight gradients**: FSDP2 starts its post-backward work when autograd computes the input gradients. Zero Bubble schedules compute the weight gradients later, so the program must control their reduction and the weight lifetime.
*   **Small collectives between nodes**: FSDP2 issues one collective per `fully_shard` unit. The d9d models make four units per layer, two of them norms. In a three-node profile, these small collectives used NCCL's low-latency protocol and lost inter-node bandwidth.
*   **Two mechanisms for one concept**: DDP swaps module classes in forward hooks (`ToLocalParallel`) and reduces with its own buckets. FSDP and HSDP run through FSDP2 with forced sum reduction. Each has its own stream ordering and its own memory behavior.

## Design Proposal

### Segments

The runtime frees the weights of one part of a stage before it gathers the next one. So the backward pass of a part must run while its weights are present, and the stage must name its parts.

This DEP introduces `StageSegment`, `ModuleSupportsSegments` and `as_segments()` in `d9d.pipelining.api`:

```python
@dataclasses.dataclass(frozen=True, slots=True)
class StageSegment:
    """One unit of a pipeline stage that runs on its own, such as a decoder layer.

    Attributes:
        name: The name of the segment, unique within the stage.
        module: The module that runs the segment and holds its parameters.
    """

    name: str
    module: nn.Module


@typing.runtime_checkable
class ModuleSupportsSegments(typing.Protocol):
    """Protocol for pipeline stage modules that describe their forward pass as a chain of segments."""

    def prepare_segment_constants(self, shared: Any) -> Any:
        """Builds the constants of one microbatch, which every segment receives. They hold no gradients."""

    def stage_segments(self) -> Sequence[StageSegment]:
        """Returns the segments of this stage, in forward order.

        After materialization, every call returns segments with the same names, order and parameter identities.
        """
```

`run_segments(module, inputs, shared)` runs the segments in order without the runtime, and models use it as `forward`. Each segment module takes the activations and the constants of one microbatch. Its parameters are the weights that the runtime gathers. A segment module can wrap the layers of the stage. The stage does not register it as a child, so checkpoint names stay the same. `as_segments(module)` turns a module without segments into one segment. A decoder takes its backbone segments through `as_segments()` and adds a segment for its heads. Its `forward` still calls the backbone module, so the hooks of the module run in eager execution.

The rules follow from the cut of the autograd graph:

*   **Activations carry all gradients**: The runtime detaches the outputs of a segment for the next one. A tensor with a gradient that reaches a later segment in any other way loses its gradient, so constants must not hold one.
*   **A boundary tensor has one producer**: Its consumers add to its gradient and run first in the backward pass. The gradient is complete when the producer runs. A tensor that a segment only passes along keeps its producer and is not cut.
*   **The list is static**: `bind()` reads the segments once, after the model is materialized and parallelized. It checks that they hold every parameter, plans the buffers from them, and the pipeline runs the same segments.

### Axis roles

The runtime must know, for each mesh dimension of a parameter, whether data parallelism gathers it, sums its gradients or leaves it to the layer. Guessing from the placement alone is wrong: a `Shard(0)` of tensor parallelism must not be gathered.

This DEP introduces `DATA_SHARD_DIMS` in `d9d.dist_context`: the mesh dimensions along which data parallelism shards parameters, `dp_shard` and `dp_cp_shard`. The runtime takes them as an explicit input:

| Placement | Mesh dimension | Weight | Gradient |
|:----------|:---------------|:-------|:---------|
| `Shard(0)` | In `DATA_SHARD_DIMS` | Gathered before the segment | Reduce-scatter |
| `Replicate` | Any | Present | All-reduce |
| `Shard` | Not in `DATA_SHARD_DIMS` | Left to the layer | Left to the layer |

DDP, FSDP and HSDP are then one implementation. Expert parallelism shards along `ep_shard`, so the experts stay with the layer. A custom strategy places parameters with `distribute_tensor()` along its own dimensions, as before. `parallelize_fsdp()` and `parallelize_hsdp()` reject a shard dimension outside `DATA_SHARD_DIMS`, or one that the mesh lacks.

### Buckets

A collective per parameter is too small between nodes, so the runtime groups parameters into buckets. One bucket uses one all-gather and one reduce-scatter. A gathered bucket holds the parameters that the same segments use, so its weights live and die together. Buckets without a gather fill up to `bucket_size_mib`. Shards are padded to equal size, so every collective has the even form.

The weights of a bucket are views of one flat buffer. The runtime frees it with `untyped_storage().resize_(0)`, so the views that autograd saved stay valid. A tied parameter has one entry in one bucket, and autograd adds up the gradients of all its uses.

### Memory

Gathered weights, gradient buffers and the temporaries of collectives live in a memory budget. A ledger counts a released buffer as busy until the GPU finishes reading it, so the host waits only when the budget runs out. The gradient shards, which live for the whole step, and the `compute_dtype` copies of parameters without a gather are not counted.

The requested budget is a target with a floor. The floor covers the gathered weights and the live gradients of the largest segment, times four for the prefetch and the collectives. The factor is a stopgap until the runtime measures the peak of a segment. A smaller request is raised to the floor. The default request is the floor, so no weights stay gathered between segments, as in FSDP. Room above the floor keeps buckets gathered for the whole step: frozen ones first, then the ones that the backward pass reads first.

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
                Values below the required minimum are raised, so the default keeps no weights between segments.
        """

    def register_stage(self, stage_index: int, module: nn.Module): ...
    def bind(self, training: bool = True): ...
    def unbind(self): ...
    def stage_segments(self, stage_index: int) -> Sequence[StageSegment]: ...

    @property
    def compute_dtype(self) -> torch.dtype | None: ...

    @contextmanager
    def compute_parameters(self, stage_index: int) -> Generator[None, None, None]:
        """Puts the unsharded parameters of a stage into its modules without their values."""

    @contextmanager
    def segment_weights(
        self, stage_index: int, segment_index: int, writes_grads: bool, next_segment: int | None = None
    ) -> Generator[None, None, None]:
        """Puts the unsharded weights of a segment into its modules for the duration of the block."""

    def reduction_plan(self, stage_index: int, final: bool) -> ReductionPlan: ...
    def reduce(self, plan: ReductionPlan, bucket_id: BucketId): ...
    def wait(self): ...
    def scale_grads(self, factor: torch.Tensor): ...
    def zero_grad(self): ...
    def invalidate_weights(self): ...
```

Every rank enters `segment_weights()` for every segment, because it issues collectives. `PipelineProgramCache` stores a `ReductionPlan` for every action that writes weight gradients. Only the last such action of a stage is final: it also sums the gradient shards over the replicas. `wait()` checks that the program reduced every bucket in a final pass.

### Windows

In a distributed job without pipeline parallelism, consecutive forward computes, and consecutive full backward computes, become window actions. A window gathers each segment once for all its microbatches, then releases its weights before the next segment. A backward window also reduces the complete gradients once, with one autograd pass per microbatch. Forward and backward windows gather separately unless the memory budget keeps the weights between them.

A window creates the compute context of each microbatch from the step handler once and enters it once per segment. A compute context can therefore be entered several times, and each entry continues where the previous one stopped. The forks of `d9d.rng` do this, so a window draws the same numbers as a stage that runs in one go.

### Compute dtype

`compute_dtype` is the dtype of every unsharded weight. A gathered weight arrives in it. A parameter without a gather gets a copy, which the first segment of each step refreshes. The gradients keep the gradient dtype of the parameter. `bind()` also converts the non-persistent floating-point buffers, such as rotary tables, to it, and persistent buffers keep their dtype. A stage describes its transfers inside `compute_parameters()`, so the receive buffers get the dtype that the unsharded parameters compute with.

`BucketId` and `ReductionPlan` are frozen dataclasses in `d9d.parallel.runtime`:

```python
@dataclasses.dataclass(frozen=True, slots=True)
class BucketId:
    """A bucket of one pipeline stage."""

    stage: int
    index: int


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class ReductionPlan:
    """The buckets that one backward pass reduces after each of its segments."""

    final: bool
    after_segment: Mapping[int, tuple[BucketId, ...]]
```

### Lifecycle

1.  `register_stage()` adds each stage module. It is allowed only while the runtime is unbound, once per stage index.
2.  `bind()` reads the segments, plans the buckets, allocates the buffers and converts the buffers of the modules. If it fails, the runtime is unbound again. Training binds in the gradient manager, and inference binds with `bind(training=False)` around its loop.
3.  A step enters `segment_weights()` for every segment on every rank and reduces each bucket after its planned segment. It then calls `wait()`, `scale_grads()`, the optimizer step and `zero_grad()`, in that order.
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
            Values below the required minimum are raised, so the default 0 keeps no weights between segments.
        bucket_size_mib: The size limit of a bucket of replicated parameters, in MiB.
    """
```

`TrainerConfig` and `InferenceConfig` get the section `parameter_runtime`. `ModelStageFactory` rejects a provider whose `parallelize_*()` calls split a tied parameter into two. `TrackedModules` takes the runtime and drops its unsharded weights before a checkpoint load.

### Verification

The implementation must keep the following measurable, with the commands and the configuration in the PR:

*   **Correctness**: Gradients and weights of DDP, FSDP, HSDP and PP with HSDP match FSDP2 and the reference within float tolerance. A checkpoint saved with FSDP2 resumes on the runtime and matches a continuous FSDP2 reference.
*   **Performance**: Steady-state step time with its spread, peak allocated memory, and the count of gathers and reductions per step, against the base commit. Include the usage reductions and the synchronization before the optimizer step. The report names both commits, the topology, the dtype, the sequence length, the global batch and the microbatch count.

## Usage

A model describes its segments:

```python
class MyModel(nn.Module, ModuleSupportsPipelining, ModuleLateInit):
    def prepare_segment_constants(self, shared: MyShared) -> MyConstants:
        return MyConstants(rope=self.rope(shared.position_ids))

    def stage_segments(self) -> list[StageSegment]:
        return [
            StageSegment(f"layer.{name}", layer)
            for name, layer in self.layers.items()
        ]

    def forward(self, inputs: MyInput, shared: MyShared) -> MyTransfer:
        return run_segments(self, inputs, shared)
```

The provider places parameters as before:

```python
def parallelize_model_stage(self, context: ParallelizeModelStageContext[MyModel]):
    mesh = context.dist_context.mesh_for(DENSE_DOMAIN)
    for layer in context.model.layers.values():
        parallelize_hsdp(layer, mesh=mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])
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
*   **Hooks**: A post-accumulate-grad hook sees the summed gradient once per step, not the local gradient of each microbatch. For a stage with explicit segments, the pipeline calls them directly, so a forward hook on that stage does not run. Hooks inside the segments run as before, and eager `forward` calls are unchanged.
*   **Sharding**: The runtime gathers only `Shard(0)` along one data shard dimension. A parameter sharded along a data shard dimension and another dimension is rejected.
*   **Compute contexts**: A window enters the compute context of a microbatch once per segment. A custom `compute_context()` must return a context that can be entered again, so a generator from `@contextlib.contextmanager` does not work there.
*   **Forks of the global generators**: `fork_rng()` and `fork_rng_child()` from DEP-0011 return contexts that can be entered again and continue their numbers.
*   **Removed modules**: `d9d.parallel.grad_sync` and `d9d.parallel.style.ToLocalParallel` are removed. `TrackedModules` moves to `d9d.loop.component.tracked_modules`, and `d9d.loop.component` still exports it.

Models without segments keep working as one segment. Training checkpoints keep their format, because every rank still saves the local shards of its `DTensor` parameters.

## Alternatives Considered

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

### A new fork per segment

A window could open a fresh compute context for each segment. Each opening would seed the generators from the same key. Every segment of a microbatch would draw the same numbers, e.g. the same dropout mask.

### A budget from the GPU memory

A budget of a fraction of the GPU memory keeps more weights gathered. At `bind()`, the runtime cannot see the peaks of the activations and the allocator, so such a default can run out of memory. The floor gives the memory of FSDP, and a larger budget is a choice of the user.

## Prerequisites

Separate PRs cover split backward (#147), the forward cache (#146), optimizer resume (#144), and tied parameters and buffers (#145). Other prerequisites are the stage computations package (#148), stateless aggregators (#149) and reusable RNG forks. Checkpoint save and load outside inference mode is a separate fix (#160).

## Deferred Work

*   **Tensor parallelism**: TP dimensions already stay with the layer. Partial gradients of replicated TP parameters, e.g. norms with sequence parallelism, need a reduction rule.
*   **Windows with pipeline parallelism**: A window delays the output of a stage that another rank waits for. It needs a schedule that plans for this delay.
*   **Tied parameters across stages**: A parameter that two stages on different ranks share needs a gradient exchange.
*   **Other shard dimensions**: The runtime gathers only `Shard(0)`.
*   **CPU offload of parameters**: FSDP2 offered `CPUOffloadPolicy`, and the runtime has none. Sleep and wake still move the whole state to host memory between steps.
