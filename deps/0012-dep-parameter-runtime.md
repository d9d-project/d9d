---
DEP: 0012
Title: Parameter Runtime
Author: Manushin Maxim
Type: Feature
---

# Parameter Runtime

## Abstract

d9d runs data parallelism with module hooks: PyTorch FSDP2 for sharded parameters and the `GradientSynchronizer` for replicated ones. The hooks decide when to gather weights and when to reduce gradients, but they cannot see the pipeline program, which already knows the order of all forward and backward work.

This DEP lets the pipeline program drive data parallelism. A model describes each pipeline stage as a static chain of segments. Around each segment, the program gathers its weights. After its backward pass, the program reduces the gradients that are complete. One `ParameterRuntime` runs DDP, FSDP and HSDP from the placements of the parameters, and the `parallelize_*` functions only place parameters.

## Motivation

*   **Gathers repeat per microbatch**: FSDP2 gathers a module in its pre-forward hook, so GPipe with $N$ microbatches gathers each layer $N$ times in the forward pass. The program runs these forward passes in a row and could gather once.
*   **The last backward pass is a guess**: The `GradientSynchronizer` counts backward passes with `set_required_accumulations()`, and d9d disables the replica all-reduce of FSDP2 to run it itself. Zero Bubble schedules split a backward pass into two actions, which a count of hooks does not see.
*   **Small collectives between nodes**: FSDP2 issues one collective per `fully_shard` unit. The d9d models make four units per layer, two of them norms. Few-MiB collectives over 24 ranks make NCCL pick its low-latency protocol, which loses most of the inter-node bandwidth.
*   **Two mechanisms for one concept**: DDP swaps module classes in forward hooks (`ToLocalParallel`) and reduces with its own buckets. FSDP and HSDP run through FSDP2 with forced sum reduction. Each has its own stream ordering and its own memory behavior.

## Design Proposal

### Segments

The runtime frees the weights of one part of a stage before it gathers the next one. So the backward pass of a part must run while its weights are present, and the stage must name its parts.

This DEP introduces `StageSegment` and `ModuleSupportsSegments` in `d9d.pipelining.api`:

```python
@dataclasses.dataclass(frozen=True, slots=True)
class StageSegment:
    """One unit of a pipeline stage that the runtime prepares and runs on its own, such as a decoder layer.

    Attributes:
        name: The name of the segment, unique within the stage.
        modules: The modules that hold the parameters of the segment.
        call: Maps the activations of the previous segment and the constants of the microbatch to the next
            activations.
    """

    name: str
    modules: tuple[nn.Module, ...]
    call: Callable[[Any, Any], Any]


@typing.runtime_checkable
class ModuleSupportsSegments(typing.Protocol):
    """Protocol for pipeline stage modules that describe their forward pass as a chain of segments."""

    def prepare_segment_constants(self, shared: Any) -> Any:
        """Builds the constants of one microbatch, which every segment receives. They hold no gradients."""

    def stage_segments(self) -> Sequence[StageSegment]:
        """Returns the segments of this stage, in forward order. The list is the same on every call."""
```

`run_segments(module, inputs, shared)` runs the segments in order without the runtime, and models use it as `forward`. A module without segments runs as one segment.

The rules follow from the cut of the autograd graph:

*   **Activations carry all gradients**: The runtime detaches the outputs of a segment for the next one. A tensor with a gradient that reaches a later segment in any other way loses its gradient, so constants must not hold one.
*   **A boundary tensor has one producer**: Its consumers add to its gradient and run first in the backward pass, so its gradient is complete when the producer runs. A tensor that a segment only passes along keeps its producer and is not cut.
*   **The list is static**: The runtime plans its buffers once, from the segments.

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

A collective per parameter is too small between nodes, so the runtime groups parameters into buckets. One bucket uses one all-gather and one reduce-scatter. A gathered bucket holds the parameters that the same segments use, so its weights live and die together. Buckets without a gather fill up to `bucket_size_mb`. Shards are padded to equal size, so every collective has the even form.

The weights of a bucket are views of one flat buffer. The runtime frees it with `untyped_storage().resize_(0)`, so the views that autograd saved stay valid. A tied parameter has one entry in one bucket, and autograd adds up the gradients of all its uses.

### Memory

Gathered weights and gradients live in a memory budget. A ledger counts a released buffer as busy until the GPU finishes reading it, so the host waits only when the budget runs out. Room above the minimum keeps buckets gathered for the whole step: frozen ones first, then the ones that the backward pass reads first.

### The runtime

This DEP introduces `ParameterRuntime` in `d9d.parallel.runtime`:

```python
class ParameterRuntime:
    """Runtime of data parallelism that the pipeline program drives."""

    def __init__(
        self,
        gather_dims: Collection[str],
        bucket_size_mb: int,
        compute_dtype: torch.dtype | None = None,
        memory_budget_bytes: int | None = None,
    ):
        """Constructs the ``ParameterRuntime`` object.

        Args:
            gather_dims: The mesh dimensions along which data parallelism shards parameters.
            bucket_size_mb: The size limit of buckets without a gather, in MiB.
            compute_dtype: The dtype of the gathered weights, or ``None`` to keep the dtype of the storage.
            memory_budget_bytes: The budget of gathered weights and gradients, or ``None`` for a quarter of the GPU.
        """

    def register_stage(self, stage_index: int, module: nn.Module): ...
    def bind(self, training: bool = True): ...
    def unbind(self): ...

    @contextmanager
    def segment_weights(
        self, stage_index: int, segment_index: int, writes_grads: bool, next_segment: int | None = None
    ) -> Generator[None, None, None]:
        """Puts the unsharded weights of a segment into its modules for the duration of the block."""

    def reduction_plan(self, stage_index: int, final: bool) -> ReductionPlan: ...
    def reduce(self, bucket_id: BucketId, final: bool): ...
    def wait(self): ...
    def scale_grads(self, factor: torch.Tensor): ...
    def zero_grad(self): ...
    def invalidate_weights(self): ...
```

Every rank enters `segment_weights()` for every segment, because it issues collectives. `PipelineProgramCache` stores a `ReductionPlan` for every action that writes weight gradients. Only the last such action of a stage is final: it also sums the gradient shards over the replicas. `wait()` checks that the program reduced every bucket in a final pass.

In a distributed job without pipeline parallelism, consecutive forward computes, and consecutive full backward computes, become window actions. A window runs each segment for all its microbatches with one gather, one autograd pass and one reduction.

`compute_dtype` gathers weights in another dtype, while the gradients keep the gradient dtype of the parameter.

### Gradients and hooks

The gradient of a sharded parameter views the gradient shard of its bucket, because `scale_grads()` and `zero_grad()` act on the shard. PyTorch lets a post-accumulate-grad hook assign a new gradient, so the runtime copies such a value back into the shard. A gradient set to `None` is an error, and `wait()` rejects a gradient that other code replaced.

Post-accumulate-grad hooks run once per step on the sharded parameter, after the final reduction, as in FSDP2. Tensor hooks run on the local gradient of each backward pass. The runtime reads the hooks at each call, so a removed hook stops at once.

### The loop

The loops build the runtime at their edges and pass it to `ModelStageFactory`, `GradientManager` and `TrackedModules`. `ParameterRuntimeConfig` configures it:

```python
class ParameterRuntimeConfig(BaseModel):
    """Configuration for the ``ParameterRuntime``, which gathers weights and reduces gradients.

    Attributes:
        compute_dtype: The dtype of the gathered weights. If ``None``, they keep the dtype of the storage.
        memory_budget_mib: The budget of gathered weights and gradients, in MiB. If ``None``, it is a quarter of the
            GPU memory.
        bucket_size_mb: The size limit of a bucket of replicated parameters, in MiB.
    """
```

`TrainerConfig` and `InferenceConfig` get the section `parameter_runtime`. `ModelStageFactory` rejects a provider whose `parallelize_*()` calls split a tied parameter into two.

## Usage

A model describes its segments:

```python
class MyModel(nn.Module, ModuleSupportsPipelining, ModuleLateInit):
    def prepare_segment_constants(self, shared: MyShared) -> MyConstants:
        return MyConstants(rope=self.rope(shared.position_ids))

    def stage_segments(self) -> list[StageSegment]:
        return [
            StageSegment(f"layer.{name}", (layer,), partial(_run_layer, layer=layer))
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

*   **Hidden states aggregators**: `BaseHiddenStatesAggregator` keeps no state. A custom aggregator implements `collect()` and `pack()` instead of `add_hidden_states()` and `pack_with_snapshot()`.
*   **Custom loops**: A loop that builds a schedule from `parallelize_*()` models creates a `ParameterRuntime`, passes it to `build_schedule()`, binds it and calls `wait()` after each step.
*   **Configs**: `gradient_manager.bucket_size_mb` moves to `parameter_runtime.bucket_size_mb`.
*   **Parallelizers**: `parallelize_fsdp()` and `parallelize_hsdp()` take no `fully_shard()` options. The memory budget replaces `reshard_after_forward`, and `compute_dtype` with `grad_dtype` replaces the mixed precision policy.
*   **Hooks**: A post-accumulate-grad hook sees the summed gradient once per step, not the local gradient of each microbatch.
*   **Removed modules**: `d9d.parallel.grad_sync` and `d9d.parallel.style.ToLocalParallel` are removed. `TrackedModules` moves to `d9d.loop.component.tracked_modules`, and `d9d.loop.component` still exports it.

Models without segments keep working as one segment. Training checkpoints keep their format, because every rank still saves the local shards of its `DTensor` parameters.

## Alternatives Considered

### Steer FSDP2 from the program

FSDP2 offers `unshard()`, `set_reshard_after_forward()` and `set_requires_gradient_sync()`. The pipeline stages of `torch.distributed.pipelining` set these flags per microbatch. The program would still need the knowledge of the runtime, DDP would stay a second mechanism, and the collectives would stay per `fully_shard` unit.

### Hooks with a counter

The `GradientSynchronizer` counts backward passes to find the last one, and DeepSpeed takes the boundary from a configured count of micro steps. The program knows the last action of a stage exactly, also with split backward, so the plan carries a `final` flag instead.

### A collective per parameter

The first prototype of this runtime issued one collective per parameter. On 3 nodes, NCCL then picked its low-latency protocol and the step got slower than with FSDP2. Megatron-FSDP and FSDP2 also group parameters for one collective.

### Swap `.data` for the gathered weights

Replacing `param.data` breaks the views that autograd saved, e.g. `weight.t()` in `F.linear`. Resizing one `Storage` keeps every view valid, as FSDP2 does.

### Infer the gather dimension

A runtime could gather along any `Shard(0)` dimension. A tensor-parallel `Shard(0)` would then be gathered without an error. The explicit `DATA_SHARD_DIMS` leaves every other `Shard` to its layer.

## Deferred Work

*   **Tensor parallelism**: TP dimensions already stay with the layer. Partial gradients of replicated TP parameters, e.g. norms with sequence parallelism, need a reduction rule.
*   **Windows with pipeline parallelism**: A window delays the output of a stage that another rank waits for, so it needs a schedule that plans for it.
*   **Tied parameters across stages**: A parameter that two stages on different ranks share needs a gradient exchange.
*   **Other shard dimensions**: The runtime gathers only `Shard(0)`.
*   **CPU offload of parameters**: FSDP2 offered `CPUOffloadPolicy`, and the runtime has none. Sleep and wake still move the whole state to host memory between steps.
