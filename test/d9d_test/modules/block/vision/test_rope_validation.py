import pytest
import torch
from d9d.module.block.vision import VisionRotaryEmbedding2D


@pytest.mark.local
def test_rejects_head_dim_not_divisible_by_4():
    with pytest.raises(ValueError, match="divisible by 4"):
        VisionRotaryEmbedding2D(head_dim=30, max_grid_side=16, spatial_merge_size=2)


@pytest.mark.local
def test_rejects_grid_exceeding_max_side():
    rope = VisionRotaryEmbedding2D(head_dim=32, max_grid_side=8, spatial_merge_size=2)
    rope.reset_parameters()

    with pytest.raises(ValueError, match="exceeds the maximum supported grid side"):
        rope(torch.tensor([[1, 16, 16]]))
