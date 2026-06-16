import os

from pydantic import TypeAdapter

from .config import (
    AnySdpaBackendConfig,
    FlashAttention2SdpaBackendConfig,
    FlashAttention4SdpaBackendConfig,
    SdpaParameters,
    TorchSdpaBackendConfig,
)
from .protocol import SdpaBackend

_ENV_VAR = "D9D_BACKEND_AUTO_SDPA"


def _auto_detect_sdpa_backend() -> AnySdpaBackendConfig:
    """Detects the appropriate SDPA backend, preferring the environment variable override.

    Returns:
        The detected SDPA backend configuration.
    """
    forced = os.environ.get(_ENV_VAR)

    if forced is not None:
        return TypeAdapter(AnySdpaBackendConfig).validate_json(forced)

    # Programmatic default: Currently we default to FlashAttention4
    return FlashAttention4SdpaBackendConfig()


def build_sdpa_backend(
    params: SdpaParameters,
    backend_config: AnySdpaBackendConfig | None,
) -> SdpaBackend:
    resolved = backend_config if backend_config is not None else _auto_detect_sdpa_backend()

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
        case _:
            raise ValueError(f"Unknown SDPA backend: {resolved}")
