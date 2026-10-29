import torch
import torch.nn.functional as F
from torch import nn

from ..config import EagerSdpaBackendConfig, SdpaParameters
from ..protocol import SdpaBackend


class EagerSdpa(nn.Module, SdpaBackend):
    """Scaled dot-product attention implemented with explicit PyTorch ops.

    This is a reference backend with no extra dependencies.
    """

    def __init__(self, config: EagerSdpaBackendConfig, params: SdpaParameters) -> None:
        """Constructs the ``EagerSdpa`` object.

        Args:
            config: Backend configuration.
            params: Structural layer parameters.
        """
        super().__init__()

        self.sinks = nn.Parameter(torch.zeros(params.num_sinks)) if params.num_sinks is not None else None
        self._window_size = params.window_size

    def _build_mask(
        self,
        seq_len: int,
        is_causal: bool,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Builds a boolean mask of disallowed positions.

        Returns:
            A boolean mask where ``True`` marks masked-out positions, or ``None`` if neither causal masking nor a
            sliding window is requested. Shape: ``(seq_len, seq_len)``.
        """
        left, right = self._window_size
        has_window = left is not None or right is not None

        if not is_causal and not has_window:
            return None

        row = torch.arange(seq_len, device=device).unsqueeze(1)
        col = torch.arange(seq_len, device=device).unsqueeze(0)
        diff = row - col

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        if is_causal:
            mask = mask | (diff < 0)

        if left is not None:
            mask = mask | (diff > left)

        if right is not None:
            mask = mask | (-diff > right)

        return mask

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor:
        batch, seq_len, num_q_heads, _ = query_states.shape
        num_kv_heads = key_states.shape[2]
        groups = num_q_heads // num_kv_heads

        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        query = query_states.transpose(1, 2)
        key = key_states.transpose(1, 2).repeat_interleave(groups, dim=1)
        value = value_states.transpose(1, 2).repeat_interleave(groups, dim=1)

        logits = torch.matmul(query, key.transpose(2, 3)) * scale

        mask = self._build_mask(seq_len, is_causal, query.device)
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

        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        return out.transpose(1, 2).contiguous()
