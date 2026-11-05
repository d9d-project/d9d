# Horizontal Parallelism

## About

The `d9d.parallel` package provides functions that distribute modules across device meshes. These strategies are "horizontal": they split work within one pipeline stage. Pipeline parallelism is "vertical": it splits the model into stages of layers.

## Design

### DTensor-First

d9d requires every trainable parameter in a distributed run to be a `torch.distributed.tensor.DTensor`. This keeps the rest of the system simple:

*   **Checkpointing**: The checkpointing engine does not need to know the parallel strategy. It reads `DTensor.placements` to decide how to gather, deduplicate and save each tensor.
*   **Data parallelism**: The [Parameter Runtime](./runtime.md) reads the placements to find which dimensions to gather along and which to sum the gradients over.

The functions in this package only place parameters. They install no hooks and do not change how a module runs. The `ParameterRuntime` gathers the weights and reduces the gradients from the pipeline program. The training and inference loops create it. Without the loops, build it yourself (see [Parameter Runtime](./runtime.md#usage)).

### Composition over Monoliths

d9d does not use monolithic wrappers like `torch.nn.parallel.DistributedDataParallel` (DDP). DDP takes ownership of the whole model execution. Instead, d9d builds on PyTorch's `parallelize_module` API, which lets you choose a strategy for each submodule:

*   Layer A can use tensor parallelism (row-wise or column-wise).
*   Layer B, such as a router, can use replicate parallelism.
*   Layer C, such as an MoE layer, can use expert parallelism.

Data parallelism is one more placement ("Replicate") in this system, so all strategies share one interface.

## Strategies

### Replicate Parallelism

`parallelize_replicate` replicates parameters across the mesh. Use it for data parallelism or context parallelism. The runtime sums the gradients across the mesh.

Inside a segment, the module sees its parameters as plain local `torch.Tensor` objects. Standard PyTorch operations and custom kernels therefore run without changes. Outside a segment, the parameters and the state dict hold `DTensor` objects.

### Expert Parallelism (MoE)

`parallelize_expert_parallel` applies expert parallelism to an `MoELayer`. It lives in `d9d.nn.moe` next to the layer, because it works only for that layer:

1.  It shards the experts (the `GroupedLinear` weights) along the `ep_shard` mesh dimension, so each GPU holds a subset of experts. Along `ep_replicate`, the experts are replicated. `ep_shard` is not a data shard dimension, so the runtime never gathers the experts: the layer sends the tokens to them.
2.  It replicates the router and the shared expert, if any, across the whole mesh.
3.  It switches the token dispatcher of the layer to [DeepEP](https://github.com/deepseek-ai/DeepEP). This requires the `d9d[backend-expert-parallel-deepep]` extra.

### Fully Sharded Data Parallel (FSDP)

`parallelize_fsdp` shards each parameter with `Shard(0)` over a 1D mesh. The mesh dimension must be a data shard dimension (`DATA_SHARD_DIMS`: `dp_shard` or `dp_cp_shard`). The runtime gathers the parameters before the segments that use them and reduce-scatters their gradients. Gradients are summed, not averaged, because d9d normalizes gradients itself. For a multi-dimensional mesh, use `parallelize_hsdp`.

### Hybrid Sharded Data Parallel (HSDP)

`parallelize_hsdp` combines full sharding with replicate parallelism. It takes a multi-dimensional mesh and a `shard_dim`, which must be a data shard dimension. It shards along `shard_dim` and replicates along all other dimensions. It skips dimensions of size 1.

### Custom Strategies

A custom strategy is a function that turns parameters into `DTensor`s with its own placements, e.g. with `distribute_parameters()` or `distribute_tensor()`. The runtime handles the placements by their roles: it sums gradients along `Replicate` dimensions and leaves a `Shard` along any dimension outside `DATA_SHARD_DIMS` to the layer. The layer then computes with its local shard and communicates by itself, as expert parallelism does.

## Usage

The examples below use the mesh domains of [Distributed Context](../dist_context/index.md). `MyCustomLayer` stands for your own module.

### Replicate Parallelism

```python
from d9d.dist_context import DENSE_DOMAIN, DistributedContext
from d9d.parallel import parallelize_replicate

ctx: DistributedContext = ...

# Dimensions: pp, dp_replicate, dp_cp_shard, cp_replicate, tp.
dense_mesh = ctx.mesh_for(DENSE_DOMAIN)

model = MyCustomLayer(...)

parallelize_replicate(model, dense_mesh["dp_replicate", "cp_replicate"])
```

### Expert Parallelism

```python
from d9d.dist_context import EXPERT_DOMAIN, DistributedContext
from d9d.nn.moe import MoELayer, parallelize_expert_parallel

ctx: DistributedContext = ...

# Dimensions: pp, ep_replicate, ep_shard.
expert_mesh = ctx.mesh_for(EXPERT_DOMAIN)

model = MoELayer(...)

parallelize_expert_parallel(
    model,
    mesh_experts=expert_mesh["ep_replicate", "ep_shard"],
    expert_shard_dim="ep_shard",
)
```

### FSDP

```python
from d9d.dist_context import DENSE_DOMAIN, DistributedContext
from d9d.parallel import parallelize_fsdp

ctx: DistributedContext = ...

dense_mesh = ctx.mesh_for(DENSE_DOMAIN)

model = MyCustomLayer(...)

parallelize_fsdp(model, mesh=dense_mesh["dp_cp_shard"])
```

### HSDP

```python
from d9d.dist_context import DENSE_DOMAIN, DistributedContext
from d9d.parallel import parallelize_hsdp

ctx: DistributedContext = ...

# Dimensions: pp, dp_replicate, dp_cp_shard, cp_replicate, tp.
dense_mesh = ctx.mesh_for(DENSE_DOMAIN)

model = MyCustomLayer(...)

parallelize_hsdp(
    model,
    mesh=dense_mesh["dp_replicate", "dp_cp_shard", "cp_replicate"],
    shard_dim="dp_cp_shard",
)
```

## API Reference

::: d9d.parallel
