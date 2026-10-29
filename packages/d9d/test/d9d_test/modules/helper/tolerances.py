from dataclasses import dataclass
from typing import Final

import torch


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass(frozen=True)
class GradTolerance:
    tol_angle: float
    tol_norm_abs: float
    tol_norm_rel: float


# Forward tolerances follow torch.testing.assert_close guidance for exact dtypes
# Reference: https://pytorch.org/docs/stable/testing.html#torch.testing.assert_close
_FORWARD_TOLERANCE_BY_DTYPE: Final[dict[torch.dtype, Tolerance]] = {
    torch.float32: Tolerance(atol=1e-5, rtol=1e-5),
    torch.float64: Tolerance(atol=1e-10, rtol=1e-10),
    torch.float16: Tolerance(atol=2e-3, rtol=2e-2),
    torch.bfloat16: Tolerance(atol=1e-3, rtol=1e-2),
}

# Gradient-distance tolerances inherit the angle/norm check
_GRAD_TOLERANCE_BY_DTYPE: Final[dict[torch.dtype, GradTolerance]] = {
    torch.float32: GradTolerance(tol_angle=0.01, tol_norm_abs=3e-4, tol_norm_rel=0.05),
    torch.float64: GradTolerance(tol_angle=0.01, tol_norm_abs=1e-6, tol_norm_rel=0.01),
    torch.float16: GradTolerance(tol_angle=0.01, tol_norm_abs=3e-3, tol_norm_rel=0.1),
    torch.bfloat16: GradTolerance(tol_angle=0.01, tol_norm_abs=3e-4, tol_norm_rel=0.05),
}


def forward_tolerance_for(dtype: torch.dtype) -> Tolerance:
    return _FORWARD_TOLERANCE_BY_DTYPE.get(dtype, Tolerance(atol=1e-5, rtol=1e-5))


def grad_tolerance_for(dtype: torch.dtype) -> GradTolerance:
    return _GRAD_TOLERANCE_BY_DTYPE[dtype]
