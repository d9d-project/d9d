import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard


@torch.no_grad()
def sync_grads_manually(module: nn.Module):
    for param in module.parameters():
        grad = param.grad
        assert isinstance(grad, DTensor)
        sync_groups = []
        for i, placement in enumerate(grad.placements):
            if isinstance(placement, Replicate):
                sync_groups.append(grad.device_mesh.get_group(i))
            elif isinstance(placement, Shard):
                pass
            else:
                raise ValueError("Unknown placement")
        for group in sync_groups:
            dist.all_reduce(grad.to_local(), op=dist.ReduceOp.SUM, group=group)
