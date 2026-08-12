import importlib.util
import os

from pydantic import TypeAdapter

from .config import (
    AnySdpaBackendConfig,
    EagerSdpaBackendConfig,
    FlashAttention2SdpaBackendConfig,
    FlashAttention4SdpaBackendConfig,
    SdpaParameters,
    TorchSdpaBackendConfig,
)
from .protocol import SdpaBackend

_ENV_VAR = "D9D_BACKEND_AUTO_SDPA"


def _auto_detect_sdpa_backend(params: SdpaParameters) -> AnySdpaBackendConfig:
    forced = os.environ.get(_ENV_VAR)
    if forced is not None:
        return TypeAdapter(AnySdpaBackendConfig).validate_json(forced)

    has_sinks = params.num_sinks is not None
    has_window = params.window_size[0] is not None or params.window_size[1] is not None
    has_fa4 = importlib.util.find_spec("flash_attn.cute") is not None
    has_fa2 = importlib.util.find_spec("flash_attn") is not None

    # FA4: no explicit mask, but supports sinks, windows and packing.
    if has_fa4 and not params.needs_attention_mask:
        return FlashAttention4SdpaBackendConfig()

    # FA2: no explicit mask and no sinks, but supports windows and packing.
    if has_fa2 and not params.needs_attention_mask and not has_sinks:
        return FlashAttention2SdpaBackendConfig()

    # Torch SDPA: supports explicit masks, but no sinks, windows or packing.
    if not has_sinks and not has_window and not params.needs_varlen:
        return TorchSdpaBackendConfig()

    # Eager: the portable fallback that serves every capability.
    return EagerSdpaBackendConfig()


def build_sdpa_backend(
    params: SdpaParameters,
    backend_config: AnySdpaBackendConfig | None,
) -> SdpaBackend:
    """Builds the selected SDPA backend module based on the provided configuration.

    If no explicit configuration is provided, it falls back to auto-detection (either from
    the `D9D_BACKEND_AUTO_SDPA` environment variable or programmatic defaults).

    The factory resolves the appropriate module implementation, passing along the backend configuration and
    structural layer parameters.

    Args:
        params: Structural layer requirements needed by the backend.
        backend_config: Explicit SDPA backend configuration, or ``None`` to auto-detect.

    Returns:
        An instantiated SDPA module implementing the SdpaBackend protocol.

    Raises:
        ValueError: If an unknown backend configuration type is encountered.
    """
    resolved = backend_config if backend_config is not None else _auto_detect_sdpa_backend(params)

    match resolved:
        case FlashAttention4SdpaBackendConfig():
            from .impl.flash4 import FlashAttention4Sdpa  # noqa: PLC0415

            return FlashAttention4Sdpa(resolved, params)
        case FlashAttention2SdpaBackendConfig():
            from .impl.flash2 import FlashAttention2Sdpa  # noqa: PLC0415

            return FlashAttention2Sdpa(resolved, params)
        case TorchSdpaBackendConfig():
            from .impl.torch_sdpa import TorchSdpa  # noqa: PLC0415

            return TorchSdpa(resolved, params)
        case EagerSdpaBackendConfig():
            from .impl.eager import EagerSdpa  # noqa: PLC0415

            return EagerSdpa(resolved, params)
        case _:
            raise ValueError(f"Unknown SDPA backend: {resolved}")
