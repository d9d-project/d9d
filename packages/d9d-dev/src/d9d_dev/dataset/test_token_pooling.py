import pytest
import torch
from d9d.dataset import TokenPoolingType, token_pooling_mask_from_attention_mask


@pytest.mark.local
@pytest.mark.parametrize(
    ("pooling_type", "attention_mask", "expected_mask"),
    [
        # check "first": should always pick the 0-th index
        (
            TokenPoolingType.first,
            [[1, 1, 1, 0], [1, 0, 0, 0]],
            [[1, 0, 0, 0], [1, 0, 0, 0]],
        ),
        # check "last": standard right-padding cases
        # Row 0: length 3 -> index 2 should be 1
        # Row 1: length 2 -> index 1 should be 1
        (
            TokenPoolingType.last,
            [[1, 1, 1, 0], [1, 1, 0, 0]],
            [[0, 0, 1, 0], [0, 1, 0, 0]],
        ),
        # check "last": full sequence (no padding)
        (
            TokenPoolingType.last,
            [[1, 1], [1, 1]],
            [[0, 1], [0, 1]],
        ),
        # check "last": single token
        (
            TokenPoolingType.last,
            [[1, 0], [1, 0]],
            [[1, 0], [1, 0]],
        ),
        # check "all": should return exact copy
        (
            TokenPoolingType.all,
            [[1, 1, 0], [1, 0, 0]],
            [[1, 1, 0], [1, 0, 0]],
        ),
    ],
)
def test_token_pooling_logic(pooling_type, attention_mask, expected_mask):
    attention_mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
    expected_mask_tensor = torch.tensor(expected_mask, dtype=torch.long)

    result = token_pooling_mask_from_attention_mask(attention_mask_tensor, pooling_type)

    assert torch.equal(result, expected_mask_tensor)

    assert result.dtype == torch.long


@pytest.mark.local
def test_token_pooling_invalid_type():
    mask = torch.ones((2, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="Unknown pooling type"):
        token_pooling_mask_from_attention_mask(mask, "invalid_type")
