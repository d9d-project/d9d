import dataclasses
import typing
from typing import Protocol

import torch


@dataclasses.dataclass
class MediaSegments:
    """A packed stream of variable-length media segments for one modality encoder.

    All media of a microbatch is concatenated along the first dimension, segment by segment, in
    the same order as the corresponding placeholder runs occur in the row-major flattened token
    sequence. An image is a segment with ``t == 1``; a video is a segment with ``t > 1``.

    Attributes:
        features: Packed pre-processed media features, shape ``(total_patches, feature_dim)``.
        grid_thw: Per-segment feature grid (temporal, height, width), shape ``(num_segments, 3)``.
    """

    features: torch.Tensor
    grid_thw: torch.Tensor


@typing.runtime_checkable
class ModalityEncoder(Protocol):
    """Protocol for modules that turn a packed media stream into sequence-aligned embeddings.

    This acts as a structural trait for modality encoder modules (e.g. a vision tower). Any
    encoder must provide a ``__call__`` method with this signature since PyTorch modules are
    invoked directly, plus ``reset_parameters`` so the composing model can initialize it under
    late init.
    """

    def reset_parameters(self) -> None:
        """Resets the encoder's parameters."""
        ...

    def __call__(self, media: MediaSegments) -> torch.Tensor:
        """Encodes packed media segments.

        Args:
            media: The packed media stream.

        Returns:
            Media token embeddings, shape ``(total_media_tokens, hidden_size)``, ordered
            segment-by-segment. ``total_media_tokens`` is the post-merge token count (e.g. after
            spatial patch merging) and must equal the number of placeholder positions in the
            token sequence.
        """
        ...
