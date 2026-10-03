import torch


def segment_cu_seqlens(grid_thw: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Computes cumulative attention segment lengths from media segment grids.

    Every temporal frame attends only within itself: a segment with grid ``(t, h, w)`` contributes
    ``t`` attention segments of ``h * w`` patches each.

    Args:
        grid_thw: Per-segment feature grid (temporal, height, width), shape ``(num_segments, 3)``.

    Returns:
        A tuple of the cumulative sequence lengths tensor (shape ``(num_frames + 1,)``, dtype
        int32) and the length of the longest attention segment.
    """
    frame_lens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])

    cu_seqlens = frame_lens.cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

    return cu_seqlens, int(frame_lens.max().item())
