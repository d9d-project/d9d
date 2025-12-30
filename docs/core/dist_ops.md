---
title: Distributed Operations
---

# Distributed Operations

## About

The `d9d.core.dist_ops` package provides high-level wrappers around `torch.distributed` collective operations.

While PyTorch's native distributed library is powerful, it often requires significant boilerplate code - specifically the manual pre-allocation of output buffers (e.g., creating a list of empty tensors for `all_gather`).

`d9d` simplifies this by handling buffer allocation automatically. It also introduces specialized operators for handling **Variadic Shapes**, allowing ranks to exchange tensors even when they do not know the incoming tensor shapes beforehand.

## Usage Examples

### Gathering Tensors

Gathering tensors of identical shapes from all ranks. d9d automatically allocates buffers for this operation.

```python
import torch
from d9d.core.dist_context import DistributedContext
from d9d.core.dist_ops import all_gather

# Setup
ctx: DistributedContext = ...
group = ctx.mesh_regular.get_group()
rank = ctx.mesh_regular.get_rank()

# Each rank has a tensor of the same shape (e.g., [2, 2])
# but different values
local_tensor = torch.ones((2, 2), device="cuda") * rank

# Gather
gathered_tensors = all_gather(local_tensor, group=group)

for i, t in enumerate(gathered_tensors):
    print(f"From rank {i}: {t}")
```

### Gathering Tensors with Variadic Shapes

Gathering tensors where dimensions differ across ranks.

```python
import torch
from d9d.core.dist_context import DistributedContext
from d9d.core.dist_ops import all_gather_variadic_shape

# Setup
ctx: DistributedContext = ...
group = ctx.mesh_regular.get_group()
rank = ctx.mesh_regular.get_rank()

# Rank 0 has shape [1], Rank 1 has shape [2], ...
local_tensor = torch.randn((rank + 1,), device="cuda")

# Gather
# The system automatically handles the shape mismatch
gathered_tensors = all_gather_variadic_shape(local_tensor, group=group)

for i, t in enumerate(gathered_tensors):
    print(f"Rank {i} sent shape: {t.shape}")
```

### Object Communication

Sending arbitrary Python objects between ranks. These objects must be picklable.

```python
import torch.distributed as dist
from d9d.core.dist_context import DistributedContext
from d9d.core.dist_ops import all_gather_object

# Setup
ctx: DistributedContext = ...
group = ctx.mesh_regular.get_group()
rank = ctx.mesh_regular.get_rank()

# Local data
my_metadata = {
    "rank": rank,
    "the-strongest": "satoru-gojo"
}

# Gather
results = all_gather_object(my_metadata, group=group)

for data in results:
    print(f"Rank {data['rank']} sent {data}")
```

::: d9d.core.dist_ops
    options:
        show_root_heading: true
        show_root_full_path: true
