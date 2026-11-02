import importlib.util
import os

import torch
from pydantic import TypeAdapter

from .config import (
    AnyGroupedMmBackendConfig,
    EagerGroupedMmBackendConfig,
    GroupedGemmBackendConfig,
    GroupedMmParameters,
    QuackGroupedMmBackendConfig,
    TorchGroupedMmBackendConfig,
)
from .protocol import GroupedMmBackend

_ENV_VAR = "D9D_BACKEND_AUTO_GROUPED_MM"

# QuACK ships GEMM kernels for sm90 and newer only.
_QUACK_MIN_MAJOR = 9

# grouped_gemm builds its CUTLASS grouped kernels for sm80. On Hopper it loops over per-expert cuBLAS calls and loses
# to the fused backends.
_GROUPED_GEMM_MAJOR = 8

# torch.nn.functional.grouped_mm has fused kernels (CUTLASS, no host sync) only on sm90 and sm100, and only for bf16.
# PyTorch's _grouped_mm_cuda takes its fast path under scaled_mm_arch_allowed(sm90_only=True, sm100_only=True). On any
# other GPU (sm80, sm110, sm120, ...) it runs _grouped_mm_fallback, which copies the offsets to the host and loops over
# the groups with one matmul each on every call. That is several host syncs per MoE layer, worse than the eager backend.
_TORCH_GROUPED_MM_FUSED_MAJORS = (9, 10)


def _auto_detect_grouped_mm_backend() -> AnyGroupedMmBackendConfig:
    forced = os.environ.get(_ENV_VAR)

    if forced is not None:
        return TypeAdapter(AnyGroupedMmBackendConfig).validate_json(forced)

    if not torch.cuda.is_available():
        return EagerGroupedMmBackendConfig()

    major = torch.cuda.get_device_capability()[0]

    if major >= _QUACK_MIN_MAJOR and importlib.util.find_spec("quack") is not None:
        return QuackGroupedMmBackendConfig()

    if major == _GROUPED_GEMM_MAJOR and importlib.util.find_spec("grouped_gemm") is not None:
        return GroupedGemmBackendConfig()

    if major in _TORCH_GROUPED_MM_FUSED_MAJORS:
        return TorchGroupedMmBackendConfig()

    return EagerGroupedMmBackendConfig()


def build_grouped_mm_backend(
    params: GroupedMmParameters, backend_config: AnyGroupedMmBackendConfig | None
) -> GroupedMmBackend:
    """Builds the grouped GEMM backend for the given configuration.

    If ``backend_config`` is ``None``, the backend config is read from the ``D9D_BACKEND_AUTO_GROUPED_MM``
    environment variable (a JSON backend config). If the variable is not set, the factory picks QuACK on sm90+ if it
    is installed, ``grouped_gemm`` on sm80 if it is installed, PyTorch ``grouped_mm`` on sm90 and sm100, and the eager
    backend otherwise.

    Args:
        params: Structural layer parameters the backend must support.
        backend_config: Explicit grouped GEMM backend configuration, or ``None`` to auto-detect.

    Returns:
        The grouped GEMM backend.

    Raises:
        ValueError: If the backend configuration type is unknown.
    """
    resolved = backend_config if backend_config is not None else _auto_detect_grouped_mm_backend()

    match resolved:
        case TorchGroupedMmBackendConfig():
            from .impl.torch_grouped_mm import TorchGroupedMm  # noqa: PLC0415

            return TorchGroupedMm(params)
        case QuackGroupedMmBackendConfig():
            from .impl.quack import QuackGroupedMm  # noqa: PLC0415

            return QuackGroupedMm(params)
        case GroupedGemmBackendConfig():
            from .impl.grouped_gemm import GroupedGemm  # noqa: PLC0415

            return GroupedGemm()
        case EagerGroupedMmBackendConfig():
            from .impl.eager import EagerGroupedMm  # noqa: PLC0415

            return EagerGroupedMm()
        case _:
            raise ValueError(f"Unknown grouped GEMM backend config ({resolved}).")
