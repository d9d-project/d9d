# Parameter Runtime

## About

The `ParameterRuntime` runs data parallelism for the pipeline program: DDP, FSDP and HSDP. Before a segment of a model runs, it gathers the full weights of that segment. After the backward pass of the segment, it reduces the gradients that are complete. The pipeline program decides these points in advance, so no module hooks are involved. The training and inference loops create one runtime per job.

## Segments

A pipeline stage is a chain of segments. A segment is a part of the forward pass whose parameters the runtime gathers and frees together, such as a decoder layer. The autograd graph is cut between segments, so each segment runs its own backward pass while its weights are present.

A segment receives two values:

*   **Activations**: Values with gradients. Each segment receives the activations of the previous one and returns new ones.
*   **Constants**: Values without gradients that every segment of a microbatch receives, such as RoPE embeddings.

A model that describes no segments runs as one segment. FSDP then gathers all weights of the stage at once. [Model Design](../models/model_design.md#segments) shows how to describe segments.

## Axis Roles

The placements of a parameter and the data shard dimensions of the job decide what the runtime does along each mesh dimension:

| Placement | Mesh dimension | Weight | Gradient |
|:----------|:---------------|:-------|:---------|
| `Shard(0)` | In `DATA_SHARD_DIMS` | Gathered before the segment | Reduce-scatter |
| `Replicate` | Any | Present on every rank | All-reduce |
| `Shard` | Not in `DATA_SHARD_DIMS` | Left to the layer | Left to the layer |

`DATA_SHARD_DIMS` lists the mesh dimensions that data parallelism shards along: `dp_shard` and `dp_cp_shard`. `parallelize_fsdp()` and `parallelize_hsdp()` accept only these dimensions. A `Shard` along any other dimension belongs to the layer, as with the experts of an `MoELayer` along `ep_shard`. A custom strategy can therefore place parameters with `distribute_tensor()` along its own dimensions, and the runtime only sums their gradients along the `Replicate` dimensions.

## Memory

Gathered weights and gradients live in a memory budget. The minimum budget holds the largest segment, the prefetch of the next one and the temporary buffers of collectives. Room above the minimum keeps weights gathered for the whole step, so the backward pass does not gather them again. Frozen weights are kept first, because they stay valid between steps. Then the runtime keeps the weights that the backward pass reads first. Inference keeps weights in the same way.

The default budget is a quarter of the GPU memory. A budget below the minimum is raised to the minimum.

## Precision

`compute_dtype` sets the dtype of the gathered weights. With `compute_dtype: bfloat16` over fp32 parameters, segments compute in bf16 and gradients accumulate in fp32. The gradient dtype follows `gradient_manager.grad_dtype`, or the dtype of the parameter. Non-persistent floating-point buffers, such as RoPE caches, take the compute dtype. Persistent buffers are training state, such as the expert bias of a router, so they keep their dtype.

PyTorch `AdamW` needs gradients in the dtype of the parameters. For bf16 compute with fp32 gradients, keep fp32 parameters and set `compute_dtype`. To keep bf16 parameters with fp32 gradients, use a [stochastic optimizer](../optim/stochastic.md).

## Microbatch Windows

In a distributed job without pipeline parallelism, the program runs consecutive forward passes, and consecutive full backward passes, as one window. Each segment gathers its weights once for the window and reduces its gradients once. With GPipe, a step then gathers each segment once in the forward pass and once in the backward pass, whatever the number of microbatches.

## Hooks

*   **Tensor hooks**: A hook from `param.register_hook()` runs on the gradient of each backward pass of this rank, before any reduction. It receives a plain tensor.
*   **Post-accumulate-grad hooks**: A hook from `param.register_post_accumulate_grad_hook()` runs once per step, after the final reduction, as in FSDP2. It receives the parameter of the module, and its `.grad` holds the gradient summed over all ranks. The loop divides the gradients by the loss weight later.

The runtime reads the hooks of a parameter at each call, so a hook that you add or remove takes effect at once.

The gradient of a parameter is a view of a runtime buffer, which the runtime scales and clears. A hook can change it in place or assign a new tensor: the runtime copies the new value back into its buffer. A hook must not set the gradient to `None`. Code outside the hooks must change gradients in place, and `wait()` rejects a gradient that was replaced.

## Limits

*   Windows need a distributed job without pipeline parallelism. With pipeline parallelism, each microbatch gathers the weights of each segment. A job on one GPU gathers nothing, and runs its microbatches one by one.
*   A tied parameter, such as an embedding that the LM head reuses, must be on one pipeline stage. Place it with one `parallelize_*` call, which keeps the tie.
*   A gathered parameter must be sharded along dimension 0 of one data shard dimension. It cannot also be sharded along another mesh dimension.
*   The runtime has no CPU offload of parameters. [Sleep and wake](../loop/train.md#sleep-and-wake-colocated-rl) moves the whole state to host memory between steps.
*   With `compute_dtype`, every floating-point tensor that crosses a pipeline stage boundary must have the compute dtype. The receiving stage allocates its buffers in this dtype.
*   DeepEP supports bf16 hidden states only.

## Usage

```yaml
trainer:
  parameter_runtime:
    compute_dtype: bfloat16  # Optional: gather fp32 parameters as bf16.
    memory_budget_mib: 16384  # Optional. The default is a quarter of the GPU memory.
    bucket_size_mb: 32  # The size of a group of replicated parameters that reduces with one all-reduce.
  gradient_manager:
    grad_dtype: float32
```

`InferenceConfig` has the same `parameter_runtime` section. The model provider places parameters with the `parallelize_*` functions and needs no other code:

```python
parallelize_hsdp(context.model, mesh=mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])
```

Outside the loop, build a runtime, pass it to `build_schedule()` and bind it before the first step:

```python
from d9d.dist_context import DATA_SHARD_DIMS
from d9d.parallel.runtime import ParameterRuntime

runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mb=32)
schedule_info, modules = build_schedule(pp_mesh, program_builder, model_provider, runtime=runtime)
runtime.bind()

for step in steps:
    runtime.zero_grad()
    schedule_info.schedule.step(inputs_microbatches=inputs, shared_microbatches=shared, callback=loss_fn)
    runtime.wait()
    optimizer.step()
```

[Parameter Runtime Internals](./runtime_internals.md) explains how buckets, collectives and the memory budget work.

## API Reference

::: d9d.parallel.runtime

::: d9d.pipelining.api.segments
