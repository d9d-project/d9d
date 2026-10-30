import dataclasses

import torch


@dataclasses.dataclass(frozen=True, slots=True)
class TensorSpec:
    """Metadata of a tensor that is not allocated on any device.

    Attributes:
        shape: The tensor shape.
        dtype: The tensor data type.
        layout: The tensor memory layout. Defaults to ``torch.strided``.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    layout: torch.layout = torch.strided
