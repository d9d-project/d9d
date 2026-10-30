import torch
import torch.nn.functional as F

from .config import GroupedMmParameters


class GroupedMmPadding:
    """Zero-pads the feature dimensions to the stride alignment a grouped GEMM kernel needs.

    The alignment is given in bytes, so the number of padded elements depends on the dtype of the operands. Padding is
    transparent: zero input features contribute nothing to the product, and the zero output features are cut off.
    """

    def __init__(self, params: GroupedMmParameters, alignment_bytes: int):
        """Constructs the ``GroupedMmPadding`` object.

        Args:
            params: Structural layer parameters.
            alignment_bytes: Stride alignment the kernel needs, in bytes.

        Raises:
            ValueError: If ``alignment_bytes`` is not positive.
        """
        if alignment_bytes <= 0:
            raise ValueError(f"alignment_bytes ({alignment_bytes}) must be positive.")

        self._in_features = params.in_features
        self._out_features = params.out_features
        self._alignment_bytes = alignment_bytes

    def _pad_size(self, features: int, element_size: int) -> int:
        return (-features) % (self._alignment_bytes // element_size)

    def pad(self, x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pads the operands to aligned feature dimensions.

        Args:
            x: Rows of all groups. Shape: ``(total_rows, in_features)``.
            weight: Per-group weights. Shape: ``(n_groups, in_features, out_features)``.

        Returns:
            The padded rows and weights.
        """
        pad_in = self._pad_size(self._in_features, weight.element_size())
        pad_out = self._pad_size(self._out_features, weight.element_size())
        if pad_in:
            x = F.pad(x, (0, pad_in))
        if pad_in or pad_out:
            weight = F.pad(weight, (0, pad_out, 0, pad_in))
        return x, weight

    def unpad(self, out: torch.Tensor) -> torch.Tensor:
        """Cuts the padded output features off.

        Args:
            out: The product of the padded operands. Shape: ``(total_rows, padded_out_features)``.

        Returns:
            The result. Shape: ``(total_rows, out_features)``.
        """
        if self._pad_size(self._out_features, out.element_size()):
            return out[:, : self._out_features]
        return out
