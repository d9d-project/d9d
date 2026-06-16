import dataclasses
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class FlashAttention4SdpaBackendConfig(BaseModel):
    """Configuration for the Flash Attention 4 backend.

    Attributes:
        kind: Discriminator field. Always "flash_attention_4".
    """

    kind: Literal["flash_attention_4"] = "flash_attention_4"


class FlashAttention2SdpaBackendConfig(BaseModel):
    """Configuration for the Flash Attention 2 backend.

    Attributes:
        kind: Discriminator field. Always "flash_attention_2".
    """

    kind: Literal["flash_attention_2"] = "flash_attention_2"


class TorchSdpaBackendType(StrEnum):
    """Available SDPA backend implementations in PyTorch."""

    MATH = "MATH"
    FLASH_ATTENTION = "FLASH_ATTENTION"
    EFFICIENT_ATTENTION = "EFFICIENT_ATTENTION"
    CUDNN_ATTENTION = "CUDNN_ATTENTION"


class TorchSdpaBackendConfig(BaseModel):
    """Configuration for the PyTorch SDPA backend.

    Attributes:
        kind: Discriminator field. Always "torch".
        backends: A list of backends to enable during SDPA. If multiple are provided,
            PyTorch will try them in order or select the best one based on heuristics.
            If ``None``, relies on PyTorch's default behavior.
    """

    kind: Literal["torch"] = "torch"
    backends: list[TorchSdpaBackendType] | None = None


AnySdpaBackendConfig = Annotated[
    FlashAttention4SdpaBackendConfig | FlashAttention2SdpaBackendConfig | TorchSdpaBackendConfig,
    Field(discriminator="kind"),
]


@dataclasses.dataclass(kw_only=True)
class SdpaParameters:
    """Internal structural parameters passed to SDPA backend factories.

    Attributes:
        num_sinks: Number of learnable sink scalars (one per query head).
            ``None`` disables sinks and gives plain attention.
        window_size: Sliding-window size for local attention as a tuple `(left, right)`.
            ``(None, None)`` disables the window and uses full attention.
    """

    num_sinks: int | None
    window_size: tuple[int | None, int | None] = (None, None)
