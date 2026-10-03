# Horizontal Parallelism

## About

The `d9d.module.parallelism` package provides functions that distribute modules across device meshes. These strategies are "horizontal": they split work within one pipeline stage. Pipeline parallelism is "vertical": it splits the model into stages of layers.

## Design

### DTensor-First

d9d requires every trainable parameter in a distributed run to be a `torch.distributed.tensor.DTensor`. This keeps the rest of the system simple:

*   **Checkpointing**: The checkpointing engine does not need to know the parallel strategy. It reads `DTensor.placements` to decide how to gather, deduplicate and save each tensor.
*   **Gradient synchronization**: [Gradient Synchronization](../internals/grad_sync.md) reads the placements to find the mesh dimensions along which a parameter is replicated.

### Composition over Monoliths

d9d does not use monolithic wrappers like `torch.nn.parallel.DistributedDataParallel` (DDP). DDP takes ownership of the whole model execution. Instead, d9d builds on PyTorch's `parallelize_module` API, which lets you choose a strategy for each submodule:

*   Layer A can use **tensor parallelism** (row-wise or column-wise).
*   Layer B, such as a router, can use **replicate parallelism**.
*   Layer C, such as an MoE layer, can use **expert parallelism**.

Data parallelism is one more placement ("Replicate") in this system, so all strategies share one interface.

## Strategies

### Replicate Parallelism

`parallelize_replicate` replicates parameters across the mesh. Use it for data parallelism or context parallelism.

During the forward pass, the module sees its parameters as plain local `torch.Tensor` objects. Standard PyTorch operations and custom kernels therefore run without changes. Outside the forward pass, the parameters and the state dict hold `DTensor` objects.

### Expert Parallelism (MoE)

`parallelize_expert_parallel` applies expert parallelism to an `MoELayer`:

1.  It shards the experts (the `GroupedLinear` weights) along the `ep_shard` mesh dimension, so each GPU holds a subset of experts. Along `ep_replicate`, the experts are replicated.
2.  It replicates the router and the shared expert, if any, across the whole mesh.

### Fully Sharded Data Parallel (FSDP)

`parallelize_fsdp` is a thin wrapper around PyTorch's `fully_shard`. It differs from plain FSDP in two ways:

*   Plain FSDP averages gradients across the mesh. `parallelize_fsdp` sums them instead, because d9d normalizes gradients itself.
*   `parallelize_fsdp` requires a 1D mesh. For a multi-dimensional mesh, use `parallelize_hsdp`, or first apply `parallelize_replicate` to the other dimensions.

### Hybrid Sharded Data Parallel (HSDP)

`parallelize_hsdp` combines full sharding with replicate parallelism. It takes a multi-dimensional mesh and a `shard_dim`. It applies `parallelize_fsdp` along `shard_dim` and `parallelize_replicate` along all other dimensions. It skips dimensions of size 1.

## Usage

The examples below use the mesh domains of [Distributed Context](../core/dist_context.md). `MyCustomLayer` stands for your own module.

### Replicate Parallelism

```python
from d9d.core.dist_context import DENSE_DOMAIN, DistributedContext
from d9d.module.parallelism.api import parallelize_replicate

ctx: DistributedContext = ...

# Dimensions: pp, dp_replicate, dp_cp_shard, cp_replicate, tp.
dense_mesh = ctx.mesh_for(DENSE_DOMAIN)

model = MyCustomLayer(...)

parallelize_replicate(model, dense_mesh["dp_replicate", "cp_replicate"])
```

### Expert Parallelism

```python
from d9d.core.dist_context import EXPERT_DOMAIN, DistributedContext
from d9d.module.block.moe import MoELayer
from d9d.module.parallelism.api import parallelize_expert_parallel

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
from d9d.core.dist_context import DENSE_DOMAIN, DistributedContext
from d9d.module.parallelism.api import parallelize_fsdp

ctx: DistributedContext = ...

dense_mesh = ctx.mesh_for(DENSE_DOMAIN)

model = MyCustomLayer(...)

parallelize_fsdp(model, mesh=dense_mesh["dp_cp_shard"])
```

### HSDP

```python
from d9d.core.dist_context import DENSE_DOMAIN, DistributedContext
from d9d.module.parallelism.api import parallelize_hsdp

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

::: d9d.module.parallelism.api

::: d9d.module.parallelism.style
