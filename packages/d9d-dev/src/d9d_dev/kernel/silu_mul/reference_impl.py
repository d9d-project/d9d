import torch
import torch.nn.functional as F


def silu_mul_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.silu(x) * y
