import dataclasses

import torch

from d9d.module.base import MediaSegments


@dataclasses.dataclass
class MultimodalSequenceInput:
    """The inputs for a multimodal sequence transformer, fed to the first stage.

    Attributes:
        input_ids: Indices of input sequence tokens, shape ``(batch, seq)``. Media placeholder
            positions hold the model's media placeholder token ids.
        media: The packed media stream consumed by the modality encoder.
        media_token_mask: Boolean mask of placeholder positions, shape ``(batch, seq)``. The number
            of ``True`` entries must equal the number of media tokens the modality encoder
            produces for ``media``.
    """

    input_ids: torch.Tensor
    media: MediaSegments
    media_token_mask: torch.Tensor
