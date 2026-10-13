import dataclasses

import torch


@dataclasses.dataclass(kw_only=True)
class Scalar:
    """Scalar value logged to a tracker.

    Creating a ``Scalar`` raises a ``ValueError`` if ``value`` is a tensor that does not have exactly one element.

    Attributes:
        value: The value, as a Python number or a tensor with one element.
    """

    value: torch.Tensor | float

    def __post_init__(self):
        if isinstance(self.value, torch.Tensor) and self.value.numel() != 1:
            raise ValueError(f"A scalar must have one element, but has {self.value.numel()} elements.")


@dataclasses.dataclass(kw_only=True)
class Histogram:
    """Histogram logged to a tracker.

    Creating a ``Histogram`` raises a ``ValueError`` if ``counts`` is not one-dimensional.

    Attributes:
        counts: The counts, one per bin. Shape: ``(num_bins,)``.
    """

    counts: torch.Tensor

    def __post_init__(self):
        if self.counts.ndim != 1:
            raise ValueError(f"Histogram counts must be one-dimensional, but have shape {tuple(self.counts.shape)}.")


TrackerValue = Scalar | Histogram
"""A value that a tracker run can write."""
