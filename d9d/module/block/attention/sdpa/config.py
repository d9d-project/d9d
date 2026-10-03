import dataclasses
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class EagerSdpaBackendConfig(BaseModel):
    """Configuration for the eager backend.

    The eager backend implements attention with explicit PyTorch ops.

    Attributes:
        kind: Discriminator field. Always ``"eager"``.
    """

    kind: Literal["eager"] = "eager"


class FlashAttention4SdpaBackendConfig(BaseModel):
    """Configuration for the FlashAttention 4 backend.

    Requires the ``d9d[backend-sdpa-flash-attention-4]`` extra.

    Attributes:
        kind: Discriminator field. Always ``"flash_attention_4"``.
    """

    kind: Literal["flash_attention_4"] = "flash_attention_4"


class FlashAttention2SdpaBackendConfig(BaseModel):
    """Configuration for the FlashAttention 2 backend.

    Requires the ``d9d[backend-sdpa-flash-attention-2]`` extra.

    Attributes:
        kind: Discriminator field. Always ``"flash_attention_2"``.
    """

    kind: Literal["flash_attention_2"] = "flash_attention_2"


class TorchSdpaBackendType(StrEnum):
    """SDPA kernels available in PyTorch.

    Each member maps to the ``torch.nn.attention.SDPBackend`` member of the same name.

    Attributes:
        MATH: Reference implementation in plain PyTorch ops.
        FLASH_ATTENTION: FlashAttention kernel.
        EFFICIENT_ATTENTION: Memory-efficient attention kernel.
        CUDNN_ATTENTION: cuDNN attention kernel.
    """

    MATH = "MATH"
    FLASH_ATTENTION = "FLASH_ATTENTION"
    EFFICIENT_ATTENTION = "EFFICIENT_ATTENTION"
    CUDNN_ATTENTION = "CUDNN_ATTENTION"


class TorchSdpaBackendConfig(BaseModel):
    """Configuration for the PyTorch SDPA backend.

    Attributes:
        kind: Discriminator field. Always ``"torch"``.
        backends: PyTorch SDPA kernels to enable. PyTorch picks one of the enabled kernels that supports the
            inputs. If ``None``, PyTorch uses its default kernel selection.
    """

    kind: Literal["torch"] = "torch"
    backends: list[TorchSdpaBackendType] | None = None


AnySdpaBackendConfig = Annotated[
    FlashAttention4SdpaBackendConfig
    | FlashAttention2SdpaBackendConfig
    | TorchSdpaBackendConfig
    | EagerSdpaBackendConfig,
    Field(discriminator="kind"),
]


@dataclasses.dataclass(kw_only=True)
class SdpaParameters:
    """Structural parameters of an attention layer that an SDPA backend must support.

    Attributes:
        num_sinks: Number of learnable sink scalars (one per query head).
            ``None`` disables sinks and gives plain attention.
        window_size: Sliding-window size for local attention as a tuple ``(left, right)``.
            ``(None, None)`` disables the window and uses full attention.
        needs_attention_mask: Whether the layer passes an explicit attention mask to the backend at runtime.
            When ``True``, auto-detection excludes backends that cannot accept explicit masks.
    """

    num_sinks: int | None
    window_size: tuple[int | None, int | None] = (None, None)
    needs_attention_mask: bool = False
