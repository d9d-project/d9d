import pytest
import torch
from d9d.dataset import compute_multimodal_position_ids, pad_empty_media
from d9d.module.base import MediaSegments

_MERGE = 2


@pytest.mark.local
def test_position_ids_text_only_are_sequential():
    seq_len = 6
    input_ids = torch.arange(seq_len)
    mask = torch.zeros(seq_len, dtype=torch.bool)
    grid_thw = torch.empty(0, 3, dtype=torch.long)

    position_ids = compute_multimodal_position_ids(input_ids, mask, grid_thw, spatial_merge_size=_MERGE)

    expected = torch.arange(seq_len).view(1, -1).expand(3, -1)
    torch.testing.assert_close(position_ids, expected)


@pytest.mark.local
def test_position_ids_single_image():
    # text(2 tokens) + image(4x4 grid -> 2x2 media tokens) + text(2 tokens)
    grid_thw = torch.tensor([[1, 4, 4]])
    mask = torch.tensor([False, False, True, True, True, True, False, False])
    input_ids = torch.zeros(mask.shape[0], dtype=torch.long)

    position_ids = compute_multimodal_position_ids(input_ids, mask, grid_thw, spatial_merge_size=_MERGE)

    # text prefix
    torch.testing.assert_close(position_ids[:, :2], torch.arange(2).view(1, -1).expand(3, -1))
    # media block starts at position 2
    torch.testing.assert_close(position_ids[0, 2:6], torch.full((4,), 2))  # temporal constant
    torch.testing.assert_close(position_ids[1, 2:6], torch.tensor([2, 2, 3, 3]))  # rows
    torch.testing.assert_close(position_ids[2, 2:6], torch.tensor([2, 3, 2, 3]))  # cols
    # text resumes at max(2, 3) + 1 = 4
    torch.testing.assert_close(position_ids[:, 6:], torch.tensor([[4, 5], [4, 5], [4, 5]]))


@pytest.mark.local
def test_position_ids_video_splits_frames_into_blocks():
    # video with 2 frames of 4x4 grid -> 2x2 media tokens per frame, no text between frames
    grid_thw = torch.tensor([[2, 4, 4]])
    mask = torch.tensor([True] * 8 + [False])
    input_ids = torch.zeros(mask.shape[0], dtype=torch.long)

    position_ids = compute_multimodal_position_ids(input_ids, mask, grid_thw, spatial_merge_size=_MERGE)

    # frame 1 block starts at 0
    torch.testing.assert_close(position_ids[0, :4], torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(position_ids[1, :4], torch.tensor([0, 0, 1, 1]))
    torch.testing.assert_close(position_ids[2, :4], torch.tensor([0, 1, 0, 1]))
    # frame 2 is a separate block starting at 0 + max(2, 2) = 2
    torch.testing.assert_close(position_ids[0, 4:8], torch.full((4,), 2))
    torch.testing.assert_close(position_ids[1, 4:8], torch.tensor([2, 2, 3, 3]))
    torch.testing.assert_close(position_ids[2, 4:8], torch.tensor([2, 3, 2, 3]))
    # text resumes at max(3, 3) + 1 = 4
    torch.testing.assert_close(position_ids[:, 8], torch.full((3,), 4))


@pytest.mark.local
def test_position_ids_video_with_timestamp_text_between_frames():
    # Qwen3.5-style pre-split video: <t1> <frame1: 2x2 grid -> 1 token> <t2> <frame2> <text>
    grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]])
    mask = torch.tensor([False, True, False, True, False])
    input_ids = torch.zeros(mask.shape[0], dtype=torch.long)

    position_ids = compute_multimodal_position_ids(input_ids, mask, grid_thw, spatial_merge_size=_MERGE)

    # timestamp text at 0, frame1 block at 1, timestamp text at 2, frame2 block at 3, text at 4
    expected = torch.tensor([0, 1, 2, 3, 4]).view(1, -1).expand(3, -1)
    torch.testing.assert_close(position_ids, expected)


@pytest.mark.local
def test_position_ids_reject_count_mismatch():
    grid_thw = torch.tensor([[1, 4, 4]])
    mask = torch.tensor([True, True, False])
    input_ids = torch.zeros(mask.shape[0], dtype=torch.long)

    with pytest.raises(ValueError, match="does not match"):
        compute_multimodal_position_ids(input_ids, mask, grid_thw, spatial_merge_size=_MERGE)


@pytest.mark.local
def test_pad_empty_media_passthrough():
    media = MediaSegments(features=torch.randn(4, 8), grid_thw=torch.tensor([[1, 2, 2]]))

    result = pad_empty_media(media, feature_dim=8, spatial_merge_size=_MERGE)

    assert result is media


@pytest.mark.local
def test_pad_empty_media_builds_dummy():
    result = pad_empty_media(None, feature_dim=8, spatial_merge_size=_MERGE)

    assert result.features.shape == (_MERGE * _MERGE, 8)
    torch.testing.assert_close(result.grid_thw, torch.tensor([[1, _MERGE, _MERGE]]))
