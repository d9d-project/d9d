import pytest
import torch
from d9d.dataset import PaddingSide1D, pad_stack_1d


@pytest.mark.local
def test_errors():
    # Test empty items
    with pytest.raises(ValueError, match="stack 0 items"):
        pad_stack_1d([], pad_value=0, padding_side=PaddingSide1D.right)

    # Test invalid pad_to_multiple_of
    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        pad_stack_1d(
            [torch.tensor([1])],
            pad_value=0,
            padding_side=PaddingSide1D.right,
            pad_to_multiple_of=0
        )

    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        pad_stack_1d(
            [torch.tensor([1])],
            pad_value=0,
            padding_side=PaddingSide1D.right,
            pad_to_multiple_of=-1
        )


@pytest.mark.local
@pytest.mark.parametrize(
    ("input_lists", "pad_val", "side", "multiple_of", "expected_shape", "expected_data"),
    [
        # Case 1: Simple Right Padding
        # Inputs: [1, 2], [3]
        # Max len: 2. Padded: [1, 2], [3, 0]
        (
                [[1, 2], [3]],
                0,
                PaddingSide1D.right,
                None,
                (2, 2),
                [[1, 2], [3, 0]]
        ),

        # Case 2: Simple Left Padding
        # Inputs: [1, 2], [3]
        # Max len: 2. Padded: [1, 2], [0, 3]
        (
                [[1, 2], [3]],
                0,
                PaddingSide1D.left,
                None,
                (2, 2),
                [[1, 2], [0, 3]]
        ),

        # Case 3: No Padding needed (inputs equal length)
        (
                [[1, 2], [3, 4]],
                9,
                PaddingSide1D.right,
                None,
                (2, 2),
                [[1, 2], [3, 4]]
        ),

        # Case 4: Multiple of, exact match not required
        # Inputs lengths: 2, 1. Max: 2. Multiple: 4. Target len: 4.
        # Right pad. [1, 2, 0, 0], [3, 0, 0, 0]
        (
                [[1, 2], [3]],
                0,
                PaddingSide1D.right,
                4,
                (2, 4),
                [[1, 2, 0, 0], [3, 0, 0, 0]]
        ),

        # Case 5: Multiple of, exact match
        # Input lengths: 4, 3. Max: 4. Multiple: 4. Target len: 4.
        (
                [[1, 2, 3, 4], [5, 6, 7]],
                9,
                PaddingSide1D.right,
                4,
                (2, 4),
                [[1, 2, 3, 4], [5, 6, 7, 9]]
        ),

        # Case 6: Left Padding with Multiple of
        # Input lengths: 2, 1. Max: 2. Multiple: 3. Target len: 3.
        # [0, 1, 2], [0, 0, 3]
        # Logic check: max_len=2. 2%3 != 0. max_len = 2 + (3 - 2) = 3.
        # Diff for [1,2] is 1 -> [0, 1, 2]
        # Diff for [3] is 2 -> [0, 0, 3]
        (
                [[1, 2], [3]],
                0,
                PaddingSide1D.left,
                3,
                (2, 3),
                [[0, 1, 2], [0, 0, 3]]
        ),

        # Case 7: Different Pad Value
        (
                [[1], [2, 3]],
                -1,
                PaddingSide1D.right,
                None,
                (2, 2),
                [[1, -1], [2, 3]]
        )
    ],
)
def test_ok(
        input_lists: list[list[int]],
        pad_val: int,
        side: PaddingSide1D,
        multiple_of: int | None,
        expected_shape: tuple[int, int],
        expected_data: list[list[int]]
):
    items = [torch.tensor(x) for x in input_lists]

    result = pad_stack_1d(
        items,
        pad_value=pad_val,
        padding_side=side,
        pad_to_multiple_of=multiple_of
    )

    assert result.shape == expected_shape
    assert torch.equal(result, torch.tensor(expected_data))
