import torch
import torch.nn.functional as F
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.attention.sdpa import AnySdpaBackendConfig, SdpaBackend, SdpaParameters, build_sdpa_backend
from d9d.module.block.normalization import RMSNorm
from d9d.module.block.positional import RotaryEmbeddingApplicator, RotaryEmbeddingStyle


class LowRankProjection(nn.Module):
    """Implements a low-rank linear projection with an intermediate normalization layer."""

    def __init__(self, in_features: int, bottleneck: int, out_features: int, norm_eps: float):
        """Constructs the ``LowRankProjection`` object.

        Args:
            in_features: Input dimensionality.
            bottleneck: Intermediate low-rank dimensionality.
            out_features: Output dimensionality.
            norm_eps: Epsilon for the intermediate RMSNorm layer.
        """
        super().__init__()
        self.down_proj = nn.Linear(in_features, bottleneck, bias=False)
        self.norm = RMSNorm(bottleneck, eps=norm_eps)
        self.up_proj = nn.Linear(bottleneck, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the low-rank projection to the inputs.

        Args:
            x: Input tensor. Shape: ``(..., in_features)``.

        Returns:
            Projected output tensor. Shape: ``(..., out_features)``.
        """
        return self.up_proj(self.norm(self.down_proj(x)))

    def reset_parameters(self):
        """Resets module parameters."""
        self.down_proj.reset_parameters()
        self.norm.reset_parameters()
        self.up_proj.reset_parameters()


class MultiHeadLatentAttention(nn.Module, ModuleLateInit):
    """Implements Multi-Head Latent Attention (MLA) from DeepSeek-V2.

    The layer runs these steps:

    1.  Projection to the query, either direct or through a low-rank bottleneck with RMSNorm.
    2.  Down-projection to a low-rank KV latent vector and a key RoPE sub-vector shared by all heads.
    3.  RMSNorm on the KV latent vector.
    4.  Up-projection of the KV latent vector into the key content (no-RoPE) and value sub-vectors.
    5.  Rotary position embedding (RoPE) on the query and key RoPE sub-vectors only.
    6.  Concatenation of the content and rotated sub-vectors into the final query and key heads.
    7.  Scaled dot-product attention with the configured SDPA backend.
    8.  Output projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: int | None,
        qk_down_norm_eps: float,
        is_causal: bool,
        rope_style: RotaryEmbeddingStyle,
        sdpa_backend: AnySdpaBackendConfig | None = None,
    ):
        """Constructs the ``MultiHeadLatentAttention`` object.

        Args:
            hidden_size: Hidden size.
            num_attention_heads: Number of attention heads.
            qk_nope_head_dim: Per-head dimension for the content (no-RoPE) part of Q and K.
            qk_rope_head_dim: Per-head dimension for the RoPE-rotated part of Q and K.
            v_head_dim: Per-head dimension for values.
            kv_lora_rank: Rank of the KV latent compression.
            q_lora_rank: Rank of the Q low-rank path. If ``None``, Q is projected directly.
            qk_down_norm_eps: Epsilon for the RMSNorm applied to the KV and Q latent representations.
            is_causal: Whether to apply a causal mask.
            rope_style: Rotary embedding layout style.
            sdpa_backend: Configuration for the scaled dot-product attention backend. If ``None``,
                the backend is auto-detected by ``build_sdpa_backend``.

        Raises:
            ValueError: If ``v_head_dim`` exceeds ``qk_nope_head_dim + qk_rope_head_dim``.
        """
        super().__init__()

        self._n_heads = num_attention_heads
        self._qk_nope_head_dim = qk_nope_head_dim
        self._qk_rope_head_dim = qk_rope_head_dim
        self._qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self._v_head_dim = v_head_dim
        self._kv_lora_rank = kv_lora_rank
        self._q_lora_rank = q_lora_rank
        self._scaling = self._qk_head_dim**-0.5
        self._is_causal = is_causal

        if v_head_dim > self._qk_head_dim:
            raise ValueError(
                f"v_head_dim ({v_head_dim}) must not exceed qk_head_dim ({self._qk_head_dim}). "
                f"V is zero-padded to the Q and K head size, so a larger V head size is not supported."
            )

        self.q_proj: LowRankProjection | nn.Linear
        if q_lora_rank is not None:
            self.q_proj = LowRankProjection(
                hidden_size, q_lora_rank, num_attention_heads * self._qk_head_dim, qk_down_norm_eps
            )
        else:
            self.q_proj = nn.Linear(hidden_size, num_attention_heads * self._qk_head_dim, bias=False)

        self.kv_down_proj = nn.Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
        )
        self.kv_down_norm = RMSNorm(kv_lora_rank, eps=qk_down_norm_eps)
        self.kv_up_proj = nn.Linear(
            kv_lora_rank,
            num_attention_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)

        self.rope = RotaryEmbeddingApplicator(style=rope_style)
        self.kernel: SdpaBackend = build_sdpa_backend(
            SdpaParameters(
                num_sinks=None,
                window_size=(None, None),
                needs_attention_mask=False,
            ),
            backend_config=sdpa_backend,
        )

    @property
    def q_lora_rank(self) -> int | None:
        """Rank of the Q low-rank path, or ``None`` if Q is projected directly."""
        return self._q_lora_rank

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Computes Multi-Head Latent Attention.

        Args:
            hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional attention mask.
            position_embeddings: Tuple ``(cos, sin)`` for the RoPE sub-vectors.
                Each tensor shape: ``(batch, seq_len, qk_rope_head_dim)``.

        Returns:
            Output tensor. Shape: ``(batch, seq_len, hidden_size)``.
        """
        b, s, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.q_proj(hidden_states)
        q = q.view(b, s, self._n_heads, self._qk_head_dim)
        q_nope, q_rope = q.split([self._qk_nope_head_dim, self._qk_rope_head_dim], dim=-1)
        q_rope, _ = self.rope(q_rope, q_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)

        kv = self.kv_down_proj(hidden_states)
        c_kv, k_rope = kv.split([self._kv_lora_rank, self._qk_rope_head_dim], dim=-1)
        c_kv = self.kv_down_norm(c_kv)
        kv_expanded = self.kv_up_proj(c_kv)
        kv_expanded = kv_expanded.view(b, s, self._n_heads, self._qk_nope_head_dim + self._v_head_dim)
        k_nope, v = kv_expanded.split([self._qk_nope_head_dim, self._v_head_dim], dim=-1)

        # k_rope is shared across all heads (MQA-style). expand() does not copy, so contiguous()
        # materializes the tensor before RoPE.
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self._n_heads, -1).contiguous()
        _, k_rope = self.rope(k_rope, k_rope, cos, sin)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Some SDPA backends (FlashAttention) require Q, K and V to share head_dim, but v_head_dim can be smaller.
        # Zero-padding V does not change the result: softmax(QK^T) · [V|0] = [result|0].
        pad_size = self._qk_head_dim - self._v_head_dim
        if pad_size > 0:
            v = F.pad(v, (0, pad_size))

        out = self.kernel(
            q,
            k,
            v,
            attention_mask=attention_mask,
            is_causal=self._is_causal,
            scale=self._scaling,
        )
        if pad_size > 0:
            out = out[..., : self._v_head_dim]

        out = out.reshape(b, s, self._n_heads * self._v_head_dim)
        return self.o_proj(out)

    def reset_parameters(self):
        """Resets module parameters."""
        self.q_proj.reset_parameters()
        self.kv_down_proj.reset_parameters()
        self.kv_down_norm.reset_parameters()
        self.kv_up_proj.reset_parameters()
        self.o_proj.reset_parameters()
