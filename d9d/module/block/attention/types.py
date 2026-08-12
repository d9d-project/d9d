import dataclasses

import torch


@dataclasses.dataclass
class SequencePacking:
    """Block-diagonal attention segmentation for a packed sequence.

    Several variable-length documents are concatenated into a single ``(1, total, ...)`` row and
    attended block-diagonally, so a token only attends within its own segment. This descriptor
    carries the segment boundaries that a variable-length attention implementation consumes.

    Attributes:
        cu_seqlens: Cumulative segment lengths, shape ``[num_segments + 1]``, ``int32``. Starts at 0
            and ends at ``total``; segment ``i`` spans the half-open range
            ``[cu_seqlens[i], cu_seqlens[i + 1])``.
        max_seqlen: The longest segment length in this row.
    """

    cu_seqlens: torch.Tensor
    max_seqlen: int
