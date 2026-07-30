import torch
from torch import nn

from d9d.module.base import ModuleLateInit
from d9d.module.block.attention.sdpa import AnySdpaBackendConfig
from d9d.module.block.ffn import GeluMLP

from .attention import PackedVisionAttention


class VisionBlock(nn.Module, ModuleLateInit):
    """A pre-norm residual Vision Transformer block over packed media segments.

    Applies ``LayerNorm -> PackedVisionAttention`` and ``LayerNorm -> GeluMLP``, each with a
    residual connection.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        norm_eps: float,
        sdpa_backend: AnySdpaBackendConfig | None = None,
    ):
        """Constructs a VisionBlock object.

        Args:
            hidden_size: The vision encoder hidden size.
            intermediate_size: The intermediate dim size of the FFN.
            num_attention_heads: Number of attention heads.
            norm_eps: Epsilon value for the layer normalizations.
            sdpa_backend: Explicit varlen SDPA backend configuration, or ``None`` to auto-detect.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = PackedVisionAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sdpa_backend=sdpa_backend,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = GeluMLP(hidden_size=hidden_size, intermediate_size=intermediate_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Runs the block forward pass.

        Args:
            hidden_states: Packed input tensor. Shape: ``(total_tokens, hidden_size)``.
            cu_seqlens: Cumulative segment lengths, shape ``(num_segments + 1,)``, dtype int32.
            max_seqlen: The length of the longest segment.
            position_embeddings: Tuple of ``(cos, sin)`` tensors for RoPE application.

        Returns:
            Processed tensor possessing the identical shape as the input.
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def reset_parameters(self):
        """Resets module parameters."""
        self.norm1.reset_parameters()
        self.attn.reset_parameters()
        self.norm2.reset_parameters()
        self.mlp.reset_parameters()
