import torch
import torch.nn.functional as F
from torch import nn

from ...types import SequencePacking
from ..config import EagerSdpaBackendConfig, SdpaParameters
from ..protocol import SdpaBackend


class EagerSdpa(nn.Module, SdpaBackend):
    """Scaled Dot Product Attention implemented with explicit PyTorch ops.

    This is a portable, dependency-free reference backend.

    Args:
        config: Backend configuration.
        params: Structural parameters.
    """

    def __init__(self, config: EagerSdpaBackendConfig, params: SdpaParameters) -> None:
        super().__init__()

        self.sinks = nn.Parameter(torch.zeros(params.num_sinks)) if params.num_sinks is not None else None
        self._window_size = params.window_size

    def _build_mask(
        self,
        seq_len: int,
        is_causal: bool,
        packing: SequencePacking | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Builds a boolean mask of disallowed positions, shape ``(seq_len, seq_len)``.

        ``True`` marks positions that must be masked out:

        - future positions, when ``is_causal`` is set;
        - positions outside the sliding window, when a window is configured;
        - keys in a different packed segment (block-diagonal attention), when ``packing`` is set.

        This dense ``(seq_len, seq_len)`` mask is the O(seq_len^2) correctness fallback for sequence
        packing; the flash varlen backend is the efficient path.

        Returns:
            A boolean mask tensor, or ``None`` when no masking is requested.
        """
        left, right = self._window_size
        has_window = left is not None or right is not None

        if not is_causal and not has_window and packing is None:
            return None

        positions = torch.arange(seq_len, device=device)
        row = positions.unsqueeze(1)
        col = positions.unsqueeze(0)
        diff = row - col

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        if is_causal:
            mask = mask | (diff < 0)

        if left is not None:
            mask = mask | (diff > left)

        if right is not None:
            mask = mask | (-diff > right)

        if packing is not None:
            boundaries = packing.cu_seqlens[1:-1].to(device)
            segment_id = torch.bucketize(positions, boundaries, right=True)
            mask = mask | (segment_id.unsqueeze(1) != segment_id.unsqueeze(0))

        return mask

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        packing: SequencePacking | None,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        batch, seq_len, num_q_heads, _ = query_states.shape
        num_kv_heads = key_states.shape[2]
        groups = num_q_heads // num_kv_heads

        # (B, S, H, D) -> (B, H, S, D)
        query = query_states.transpose(1, 2)
        key = key_states.transpose(1, 2).repeat_interleave(groups, dim=1)
        value = value_states.transpose(1, 2).repeat_interleave(groups, dim=1)

        logits = torch.matmul(query, key.transpose(2, 3)) * scale

        mask = self._build_mask(seq_len, is_causal, packing, query.device)
        if mask is not None:
            logits = logits.masked_fill(mask[None, None, :], float("-inf"))

        if attention_mask is not None:
            logits = logits + attention_mask

        if self.sinks is not None:
            sink_col = self.sinks.to(logits.dtype).view(1, num_q_heads, 1, 1).expand(batch, num_q_heads, seq_len, 1)
            combined = torch.cat([logits, sink_col], dim=-1)
            probs = F.softmax(combined, dim=-1)
            scores = probs[..., :-1]
        else:
            scores = F.softmax(logits, dim=-1)

        out = torch.matmul(scores.to(value.dtype), value)

        # (B, H, S, D) -> (B, S, H, D)
        return out.transpose(1, 2).contiguous()
