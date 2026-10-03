from typing import cast

import torch
from torch import nn

from d9d.module.base import MediaSegments, ModuleLateInit
from d9d.module.block.attention.sdpa import AnySdpaBackendConfig
from d9d.module.block.ffn import GELUMLP
from d9d.module.block.vision import (
    InterpolatedPositionEmbedding,
    PackedVisionAttention,
    PatchEmbedding,
    SpatialPatchMerger,
    VisionRotaryEmbedding2D,
    segment_cu_seqlens,
)

from .params import Qwen3p5VisionParameters


class Qwen3p5VisionLayer(nn.Module, ModuleLateInit):
    """Implements a Qwen3.5 vision transformer layer over packed media segments.

    This layer consists of bidirectional packed vision attention followed by a GELU MLP block,
    with pre-LayerNorm applied before each sub-layer and residual connections around both.
    """

    def __init__(self, params: Qwen3p5VisionParameters, sdpa_backend: AnySdpaBackendConfig | None = None):
        """Constructs a Qwen3p5VisionLayer object.

        Args:
            params: Configuration parameters for the vision encoder.
            sdpa_backend: Explicit varlen SDPA backend configuration, or ``None`` to auto-detect.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(params.hidden_size, eps=params.norm_eps)
        self.attn = PackedVisionAttention(
            hidden_size=params.hidden_size,
            num_attention_heads=params.num_attention_heads,
            sdpa_backend=sdpa_backend,
        )
        self.norm2 = nn.LayerNorm(params.hidden_size, eps=params.norm_eps)
        self.mlp = GELUMLP(
            hidden_size=params.hidden_size,
            intermediate_size=params.intermediate_size,
            bias=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Performs the forward pass of the vision layer.

        Args:
            hidden_states: Packed input tensor. Shape: ``(total_tokens, hidden_size)``.
            cu_seqlens: Cumulative segment lengths, shape ``(num_segments + 1,)``, dtype int32.
            max_seqlen: The length of the longest segment.
            position_embeddings: Tuple of ``(cos, sin)`` tensors for RoPE application.

        Returns:
            Processed tensor possessing the identical shape as the input.
        """
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def reset_parameters(self):
        """Resets module parameters."""
        self.norm1.reset_parameters()
        self.attn.reset_parameters()
        self.norm2.reset_parameters()
        self.mlp.reset_parameters()


class Qwen3p5VisionModel(nn.Module, ModuleLateInit):
    """The Qwen3.5 vision encoder.

    Encodes a packed stream of image/video segments into media token embeddings aligned with the
    language model hidden size. Satisfies the ``ModalityEncoder`` protocol.
    """

    def __init__(self, params: Qwen3p5VisionParameters, sdpa_backend: AnySdpaBackendConfig | None = None):
        """Constructs the Qwen3p5VisionModel object.

        Args:
            params: Configuration parameters for the vision encoder.
            sdpa_backend: Explicit varlen SDPA backend configuration, or ``None`` to auto-detect.
        """
        super().__init__()

        self.patch_embed = PatchEmbedding(
            hidden_size=params.hidden_size,
            in_channels=params.in_channels,
            patch_size=params.patch_size,
            temporal_patch_size=params.temporal_patch_size,
        )
        self.pos_embed = InterpolatedPositionEmbedding(
            hidden_size=params.hidden_size,
            num_position_embeddings=params.num_position_embeddings,
            spatial_merge_size=params.spatial_merge_size,
        )
        self.rotary = VisionRotaryEmbedding2D(
            head_dim=params.hidden_size // params.num_attention_heads,
            max_grid_side=params.max_grid_side,
            spatial_merge_size=params.spatial_merge_size,
        )
        self.blocks = nn.ModuleList(
            [Qwen3p5VisionLayer(params, sdpa_backend=sdpa_backend) for _ in range(params.depth)]
        )
        self.merger = SpatialPatchMerger(
            hidden_size=params.hidden_size,
            out_hidden_size=params.out_hidden_size,
            spatial_merge_size=params.spatial_merge_size,
            norm_eps=params.norm_eps,
        )

    def forward(self, media: MediaSegments) -> torch.Tensor:
        """Encodes packed media segments.

        Args:
            media: The packed media stream.

        Returns:
            Media token embeddings, shape ``(total_media_tokens, out_hidden_size)``.
        """
        hidden_states = self.patch_embed(media.features) + self.pos_embed(media.grid_thw)

        position_embeddings = self.rotary(media.grid_thw)
        cu_seqlens, max_seqlen = segment_cu_seqlens(media.grid_thw)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
            )

        return self.merger(hidden_states)

    def reset_parameters(self):
        """Resets module parameters."""
        self.patch_embed.reset_parameters()
        self.pos_embed.reset_parameters()
        self.rotary.reset_parameters()

        for block in self.blocks:
            cast(Qwen3p5VisionLayer, block).reset_parameters()

        self.merger.reset_parameters()
