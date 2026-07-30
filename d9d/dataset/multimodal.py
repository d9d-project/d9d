import torch

from d9d.module.base import MediaSegments


def compute_multimodal_position_ids(
    input_ids: torch.Tensor,
    media_token_mask: torch.Tensor,
    grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Computes 3D (temporal, height, width) position ids for a multimodal token sequence.

    Text tokens advance all three position planes together. Each media segment occupies a 3D
    block: the temporal plane is constant per frame, and the height/width planes enumerate the
    post-merge feature grid. After a media segment, text positions continue from the maximum
    position the segment used plus one.

    This is CPU-side index arithmetic intended to be called from the collator, one sample at a
    time.

    Args:
        input_ids: Token ids for a single sample, shape ``(seq,)``.
        media_token_mask: Boolean mask of media placeholder positions, shape ``(seq,)``.
        grid_thw: Feature grids of the sample's media segments in placeholder order, shape
            ``(num_segments, 3)``.
        spatial_merge_size: The spatial merge factor of the vision encoder.

    Returns:
        Position ids, shape ``(3, seq)``.

    Raises:
        ValueError: If the placeholder count does not match the media token count implied by
            ``grid_thw``.
    """
    seq_len = input_ids.shape[0]

    mask_list = media_token_mask.tolist()
    grids = grid_thw.tolist()

    expected_media_tokens = sum(t * (h // spatial_merge_size) * (w // spatial_merge_size) for t, h, w in grids)
    actual_media_tokens = sum(mask_list)
    if expected_media_tokens != actual_media_tokens:
        raise ValueError(
            f"Number of media placeholder positions ({actual_media_tokens}) does not match the "
            f"media token count implied by grid_thw ({expected_media_tokens})."
        )

    position_ids = torch.zeros(3, seq_len, dtype=torch.long)

    grid_iter = iter(grids)
    current_pos = 0
    token_idx = 0

    while token_idx < seq_len:
        if not mask_list[token_idx]:
            position_ids[:, token_idx] = current_pos
            current_pos += 1
            token_idx += 1
            continue

        t, h, w = next(grid_iter)
        merged_h = h // spatial_merge_size
        merged_w = w // spatial_merge_size
        num_tokens = t * merged_h * merged_w

        pos_t = torch.full((num_tokens,), current_pos, dtype=torch.long)
        pos_h = current_pos + torch.arange(merged_h).repeat_interleave(merged_w).repeat(t)
        pos_w = current_pos + torch.arange(merged_w).repeat(merged_h * t)

        position_ids[0, token_idx : token_idx + num_tokens] = pos_t
        position_ids[1, token_idx : token_idx + num_tokens] = pos_h
        position_ids[2, token_idx : token_idx + num_tokens] = pos_w

        current_pos += max(merged_h, merged_w)
        token_idx += num_tokens

    return position_ids


def pad_empty_media(
    media: MediaSegments | None,
    feature_dim: int,
    spatial_merge_size: int,
) -> MediaSegments:
    """Ensures a microbatch always carries at least one media segment.

    Microbatches without media must still run the modality encoder to keep collectives (e.g. FSDP
    all-gathers) and the autograd graph structurally identical across ranks and microbatches, and
    to satisfy the requirement that all microbatches of a pack share the same ``PipelineInput``
    PyTree structure. This helper substitutes a single minimal dummy segment whose output is
    discarded by an all-``False`` media token mask.

    Args:
        media: The packed media of the microbatch, or ``None`` when it carries no media.
        feature_dim: The packed feature dimension
            (``in_channels * temporal_patch_size * patch_size * patch_size``).
        spatial_merge_size: The spatial merge factor of the vision encoder.

    Returns:
        The input ``media`` unchanged when present, else a minimal dummy ``MediaSegments``.
    """
    if media is not None:
        return media

    num_patches = spatial_merge_size * spatial_merge_size

    return MediaSegments(
        features=torch.zeros(num_patches, feature_dim),
        grid_thw=torch.tensor([[1, spatial_merge_size, spatial_merge_size]], dtype=torch.long),
    )
