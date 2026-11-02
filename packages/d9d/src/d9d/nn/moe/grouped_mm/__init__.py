"""Provides pluggable grouped GEMM backends for Mixture-of-Experts layers."""

from .config import (
    AnyGroupedMmBackendConfig,
    EagerGroupedMmBackendConfig,
    GroupedGemmBackendConfig,
    GroupedMmParameters,
    QuackGroupedMmBackendConfig,
    TorchGroupedMmBackendConfig,
)
from .factory import build_grouped_mm_backend
from .groups import ExpertTokenGroups
from .protocol import GroupedMmBackend

__all__ = [
    "AnyGroupedMmBackendConfig",
    "EagerGroupedMmBackendConfig",
    "ExpertTokenGroups",
    "GroupedGemmBackendConfig",
    "GroupedMmBackend",
    "GroupedMmParameters",
    "QuackGroupedMmBackendConfig",
    "TorchGroupedMmBackendConfig",
    "build_grouped_mm_backend",
]
