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
from .protocol import SdpaBackend, VarlenSdpaBackend

_ENV_VAR = "D9D_BACKEND_AUTO_SDPA"
_ENV_VAR_VARLEN = "D9D_BACKEND_AUTO_SDPA_VARLEN"


def _auto_detect_sdpa_backend(params: SdpaParameters) -> AnySdpaBackendConfig:
    forced = os.environ.get(_ENV_VAR)

    if forced is not None:
        return TypeAdapter(AnySdpaBackendConfig).validate_json(forced)

    has_sinks = params.num_sinks is not None
    has_window = params.window_size[0] is not None or params.window_size[1] is not None
    needs_mask = params.needs_attention_mask

    # Probing "flash_attn.cute" directly would raise ModuleNotFoundError when the parent
    # "flash_attn" package is absent, so the parent is checked first.
    has_flash = importlib.util.find_spec("flash_attn") is not None

    if not needs_mask and has_flash and importlib.util.find_spec("flash_attn.cute") is not None:
        return FlashAttention4SdpaBackendConfig()

    if not needs_mask and not has_sinks and has_flash:
        return FlashAttention2SdpaBackendConfig()

    if not has_sinks and not has_window:
        return TorchSdpaBackendConfig()

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


def _auto_detect_varlen_sdpa_backend(params: SdpaParameters) -> AnySdpaBackendConfig:
    forced = os.environ.get(_ENV_VAR_VARLEN)

    if forced is not None:
        return TypeAdapter(AnySdpaBackendConfig).validate_json(forced)

    has_sinks = params.num_sinks is not None
    needs_mask = params.needs_attention_mask

    # Probing "flash_attn.cute" directly would raise ModuleNotFoundError when the parent
    # "flash_attn" package is absent, so the parent is checked first.
    has_flash = importlib.util.find_spec("flash_attn") is not None

    if not needs_mask and not has_sinks and has_flash and importlib.util.find_spec("flash_attn.cute") is not None:
        return FlashAttention4SdpaBackendConfig()

    if not needs_mask and not has_sinks and has_flash:
        return FlashAttention2SdpaBackendConfig()

    return TorchSdpaBackendConfig()


def build_varlen_sdpa_backend(
    params: SdpaParameters,
    backend_config: AnySdpaBackendConfig | None,
) -> VarlenSdpaBackend:
    """Builds the selected variable-length SDPA backend module based on the provided configuration.

    If no explicit configuration is provided, it falls back to auto-detection (either from
    the `D9D_BACKEND_AUTO_SDPA_VARLEN` environment variable or programmatic defaults).

    The factory resolves the appropriate module implementation, passing along the backend configuration and
    structural layer parameters.

    Args:
        params: Structural layer requirements needed by the backend.
        backend_config: Explicit SDPA backend configuration, or ``None`` to auto-detect.

    Returns:
        An instantiated SDPA module implementing the VarlenSdpaBackend protocol.

    Raises:
        ValueError: If an unknown or unsupported backend configuration type is encountered.
    """
    resolved = backend_config if backend_config is not None else _auto_detect_varlen_sdpa_backend(params)

    match resolved:
        case FlashAttention4SdpaBackendConfig():
            from .impl.flash4_varlen import FlashAttention4VarlenSdpa  # noqa: PLC0415

            return FlashAttention4VarlenSdpa(resolved, params)
        case FlashAttention2SdpaBackendConfig():
            from .impl.flash2_varlen import FlashAttention2VarlenSdpa  # noqa: PLC0415

            return FlashAttention2VarlenSdpa(resolved, params)
        case TorchSdpaBackendConfig():
            from .impl.torch_sdpa_varlen import TorchVarlenSdpa  # noqa: PLC0415

            return TorchVarlenSdpa(resolved, params)
        case EagerSdpaBackendConfig():
            raise ValueError("The eager backend has no variable-length implementation; use the torch backend instead.")
        case _:
            raise ValueError(f"Unknown SDPA backend: {resolved}")
