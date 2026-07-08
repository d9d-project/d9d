import dataclasses

import torch


@dataclasses.dataclass(frozen=True, slots=True)
class TensorSpec:
    """Describes a tensor by its metadata, without allocating it on any device.

    Attributes:
        shape: The tensor shape.
        dtype: The tensor data type.
        layout: The tensor memory layout. Defaults to ``torch.strided``.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    layout: torch.layout = torch.strided
