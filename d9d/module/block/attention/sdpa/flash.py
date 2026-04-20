import torch
import torch.nn.functional as F
from torch import nn

from d9d.kernel.flash_attn import flash_attn_func

# FA4's backward preprocess kernel requires head_dim to be a multiple of this
# value; non-aligned dims trigger a CUTE predicate-shape bug.
_FA4_HDIM_ALIGN = 32


class FlashSdpa(nn.Module):
    """Scaled Dot Product Attention using Flash Attention 4.

    When ``num_sinks`` is provided, a learnable per-head sink logit is added
    to the softmax denominator (attention-sink mechanism).  This lets a
    fraction of attention mass be absorbed by the sink, effectively
    soft-gating the output without materializing an extra KV column.

    Args:
        num_sinks: Number of learnable sink scalars (one per query head).
            ``None`` (default) disables sinks and gives plain attention.
        window_size: Sliding-window size for local attention.  ``None``
            (default) disables the window and uses full attention.
    """

    def __init__(self, num_sinks: int | None = None, window_size: int | None = None) -> None:
        super().__init__()

        self.sinks = nn.Parameter(torch.zeros(num_sinks)) if num_sinks is not None else None

        if window_size is not None and window_size < 0:
            raise ValueError("`window_size` must be either `None` or a positive integer value")

        self._window_size = window_size

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        """
        Computes Scaled Dot-Product Attention.

        Args:
            query_states: Query tensor. Shape: ``(batch, seq_len, n_q_heads, head_dim)``.
            key_states: Key tensor. Shape: ``(batch, seq_len, n_kv_heads, head_dim)``.
            value_states: Value tensor. Shape: ``(batch, seq_len, n_kv_heads, head_dim)``.
            attention_mask: Unused. Present for interface compatibility.
            is_causal: If True, applies a causal mask.
            scale: Softmax scaling factor (usually ``1/sqrt(head_dim)``).

        Returns:
            Attention output. Shape: ``(batch, seq_len, n_q_heads, head_dim)``.
        """
        if self._window_size is not None and not is_causal:
            raise ValueError("Sliding window attention requires is_causal=True")

        # Pad head_dim to the next multiple of _FA4_HDIM_ALIGN so that FA4's
        # backward preprocess kernel avoids the broken predicate-masking path.
        # Zero-padding is transparent: extra dims contribute 0 to QK^T and to
        # the weighted sum over V.
        head_dim = query_states.shape[-1]
        pad = (-head_dim) % _FA4_HDIM_ALIGN
        if pad:
            query_states = F.pad(query_states, (0, pad))
            key_states = F.pad(key_states, (0, pad))
            value_states = F.pad(value_states, (0, pad))

        window = (self._window_size, 0) if self._window_size is not None else (None, None)

        out, *_ = flash_attn_func(
            query_states,
            key_states,
            value_states,
            softmax_scale=scale,
            causal=is_causal,
            window_size=window,
            learnable_sink=self.sinks,
        )

        if pad:
            out = out[..., :head_dim]

        return out
