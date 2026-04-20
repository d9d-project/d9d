"""Eager O(S²) reference for attention with learnable sink."""

import torch
import torch.nn.functional as F


def eager_sink_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor | None,
    scale: float,
    causal: bool,
    window_size: tuple[int | None, int | None],
) -> torch.Tensor:
    """Eager attention with optional sink, matching FA4's BSHD layout.

    Args:
        q: (B, S, H_q, D)
        k: (B, S, H_kv, D)
        v: (B, S, H_kv, D)
        sink: (H_q,) or None
        scale: softmax scale
        causal: apply causal mask
        window_size: (left, right) sliding window; None entries mean no limit

    Returns:
        (B, S, H_q, D)
    """
    b, s, h_q, _d = q.shape
    h_kv = k.shape[2]
    groups = h_q // h_kv

    # Transpose to (B, H, S, D) for matmul
    q_t = q.transpose(1, 2)  # (B, H_q, S, D)
    k_t = k.transpose(1, 2).repeat_interleave(groups, dim=1)  # (B, H_q, S, D)
    v_t = v.transpose(1, 2).repeat_interleave(groups, dim=1)  # (B, H_q, S, D)

    logits = torch.matmul(q_t, k_t.transpose(2, 3)) * scale  # (B, H_q, S, S)

    if causal:
        row = torch.arange(s, device=q.device)
        col = torch.arange(s, device=q.device)
        mask = col.unsqueeze(0) > row.unsqueeze(1)
        left = window_size[0]
        if left is not None:
            mask = mask | (row.unsqueeze(1) - col.unsqueeze(0) > left)
        logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    if sink is not None:
        sink_col = sink.view(1, h_q, 1, 1).expand(b, h_q, s, 1)
        combined = torch.cat([logits, sink_col], dim=-1)  # (B, H_q, S, S+1)
        probs = F.softmax(combined, dim=-1)
        scores = probs[..., :-1]  # drop sink column
    else:
        scores = F.softmax(logits, dim=-1)

    out = torch.matmul(scores, v_t)  # (B, H_q, S, D)
    return out.transpose(1, 2).contiguous()  # (B, S, H_q, D)
