import pytest
import torch
from d9d.module.block.moe.grouped_mm import GroupedMmParameters
from d9d.module.block.moe.grouped_mm.alignment import GroupedMmPadding


@pytest.mark.local
@pytest.mark.parametrize(
    ("dtype", "padded_in", "padded_out"),
    [
        # 16-byte strides: 8 elements for 2-byte dtypes, 4 for 4-byte ones
        (torch.bfloat16, 8, 8),
        (torch.float16, 8, 8),
        (torch.float32, 4, 8),
    ],
)
def test_padding_follows_dtype(dtype, padded_in, padded_out):
    padding = GroupedMmPadding(GroupedMmParameters(in_features=4, out_features=6), alignment_bytes=16)
    x = torch.randn(10, 4, dtype=dtype)
    weight = torch.randn(3, 4, 6, dtype=dtype)

    padded_x, padded_weight = padding.pad(x, weight)

    assert padded_x.shape == (10, padded_in)
    assert padded_weight.shape == (3, padded_in, padded_out)
    assert padding.unpad(torch.zeros(10, padded_out, dtype=dtype)).shape == (10, 6)


@pytest.mark.local
@pytest.mark.parametrize("alignment_bytes", [0, -16])
def test_padding_rejects_non_positive_alignment(alignment_bytes):
    with pytest.raises(ValueError, match=rf"alignment_bytes \({alignment_bytes}\) must be positive"):
        GroupedMmPadding(GroupedMmParameters(in_features=4, out_features=6), alignment_bytes=alignment_bytes)


@pytest.mark.local
def test_aligned_dims_are_not_copied():
    padding = GroupedMmPadding(GroupedMmParameters(in_features=16, out_features=8), alignment_bytes=16)
    x = torch.randn(10, 16, dtype=torch.bfloat16)
    weight = torch.randn(3, 16, 8, dtype=torch.bfloat16)
    out = torch.randn(10, 8, dtype=torch.bfloat16)

    padded_x, padded_weight = padding.pad(x, weight)

    assert padded_x is x
    assert padded_weight is weight
    assert padding.unpad(out) is out
