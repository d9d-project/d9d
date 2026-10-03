# Distributed Operations

## About

The `d9d.core.dist_ops` package wraps `torch.distributed` collective operations and allocates their output buffers for you. With plain PyTorch, you must pre-allocate the outputs yourself, for example a list of empty tensors for `all_gather`. The package also has operations for **variadic shapes**. They let ranks exchange tensors without knowing the shapes of the incoming tensors in advance.

## Usage

### Gathering Tensors

Gather tensors of the same shape from all ranks.

```python
import torch
from d9d.core.dist_context import DistributedContext, FLAT_DOMAIN
from d9d.core.dist_ops import all_gather

# Setup.
ctx: DistributedContext = ...
mesh = ctx.mesh_for(FLAT_DOMAIN)
group = mesh.get_group()
rank = mesh.get_local_rank()

# Each rank has a tensor of the same shape, but with different values.
local_tensor = torch.ones((2, 2), device="cuda") * rank

# Gather.
gathered_tensors = all_gather(local_tensor, group=group)

for i, t in enumerate(gathered_tensors):
    print(f"From rank {i}: {t}")
```

### Gathering Tensors with Variadic Shapes

Gather tensors whose shapes differ across ranks.

```python
import torch
from d9d.core.dist_context import DistributedContext, FLAT_DOMAIN
from d9d.core.dist_ops import all_gather_variadic_shape

# Setup.
ctx: DistributedContext = ...
mesh = ctx.mesh_for(FLAT_DOMAIN)
group = mesh.get_group()
rank = mesh.get_local_rank()

# Rank 0 has shape (1,), rank 1 has shape (2,), ...
local_tensor = torch.randn((rank + 1,), device="cuda")

# Gather: the shapes are exchanged first.
gathered_tensors = all_gather_variadic_shape(local_tensor, group=group)

for i, t in enumerate(gathered_tensors):
    print(f"Rank {i} sent shape: {t.shape}")
```

### Gathering Objects

Gather Python objects from all ranks. The objects must be picklable.

```python
from d9d.core.dist_context import DistributedContext, FLAT_DOMAIN
from d9d.core.dist_ops import all_gather_object

# Setup.
ctx: DistributedContext = ...
mesh = ctx.mesh_for(FLAT_DOMAIN)
group = mesh.get_group()
rank = mesh.get_local_rank()

# Local data.
my_metadata = {
    "rank": rank,
    "status": "ready"
}

# Gather.
results = all_gather_object(my_metadata, group=group)

for data in results:
    print(f"Rank {data['rank']} sent {data}")
```

## API Reference

::: d9d.core.dist_ops
