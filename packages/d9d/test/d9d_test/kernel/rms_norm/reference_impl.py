import torch
import torch.nn.functional as F
from liger_kernel.ops import LigerRMSNormFunction


def rms_norm_torch(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False
) -> torch.Tensor:
    var = x.pow(2).mean(-1, keepdim=True)
    out = x * torch.rsqrt(var + eps)

    w = weight
    if zero_centered:
        w = w + 1.0

    out = out * w
    return out


def rms_norm_torch_functional(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False
) -> torch.Tensor:
    if zero_centered:
        raise NotImplementedError()

    return F.rms_norm(x, normalized_shape=(x.shape[-1],), weight=weight, eps=eps)


def rms_norm_liger_kernel(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, zero_centered: bool = False
) -> torch.Tensor:
    if zero_centered:
        raise NotImplementedError()
    return LigerRMSNormFunction.apply(x, weight, eps)
