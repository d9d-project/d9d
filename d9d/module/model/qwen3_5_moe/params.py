from pydantic import BaseModel, Field


class Qwen3p5MoEVisionParameters(BaseModel):
    """Configuration parameters for the Qwen3.5 MoE vision encoder.

    Attributes:
        depth: The number of vision transformer blocks.
        hidden_size: The vision encoder hidden size.
        intermediate_size: Dimension of the vision feed-forward hidden state.
        num_attention_heads: Number of attention heads in the vision blocks.
        in_channels: Number of input image channels.
        patch_size: Spatial patch size (height and width).
        temporal_patch_size: Temporal patch size (number of frames per patch).
        spatial_merge_size: The spatial merge factor of the patch merger.
        out_hidden_size: The output hidden size (the language model hidden size).
        num_position_embeddings: Total number of learned absolute positions. Must be a perfect
            square.
        max_grid_side: Maximum supported feature-grid height/width for rotary caching.
        norm_eps: Epsilon value for the vision layer normalizations.
    """

    depth: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    num_position_embeddings: int
    max_grid_side: int
    norm_eps: float


class Qwen3p5MoELayerParameters(BaseModel):
    """Configuration parameters for a single Qwen3.5 MoE decoder layer.

    Attributes:
        hidden_size: Dimension of the model's hidden states.
        moe_intermediate_size: Dimension of a single routed expert's hidden state.
        shared_expert_intermediate_size: Dimension of the shared expert's hidden state.
        num_experts: Total number of routed experts in the MoE layer.
        experts_top_k: Number of experts to route tokens to.
        num_attention_heads: Number of attention heads for the query (full-attention layers).
        num_key_value_heads: Number of attention heads for key and value (full-attention layers).
        head_dim: Dimension of a single attention head (full-attention layers).
        rope_dim: The rotary dimension (partial RoPE) of a full-attention head.
        linear_num_key_heads: Number of query/key heads in linear-attention layers.
        linear_num_value_heads: Number of value heads in linear-attention layers.
        linear_key_head_dim: Dimension of a single query/key head in linear-attention layers.
        linear_value_head_dim: Dimension of a single value head in linear-attention layers.
        linear_conv_kernel_dim: Kernel size of the causal convolution in linear-attention layers.
        rms_norm_eps: Epsilon value found in the RMSNorm layers.
    """

    hidden_size: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    num_experts: int
    experts_top_k: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    rms_norm_eps: float


class Qwen3p5MoEParameters(BaseModel):
    """Configuration parameters for the Qwen3.5 Mixture-of-Experts model backbone.

    Attributes:
        layer: Configuration shared across all decoder layers.
        num_hidden_layers: The total number of decoder layers.
        full_attention_interval: Every ``full_attention_interval``-th layer (1-based) uses full
            attention; all other layers use linear attention (Gated DeltaNet).
        rope_base: Base value for RoPE frequency calculation.
        mrope_section: Number of frequency pairs allocated to the (temporal, height, width)
            position planes. Must sum to ``layer.rope_dim // 2``.
        max_position_ids: Maximum sequence length.
        split_vocab_size: A dictionary mapping vocabulary segment names to their sizes.
        split_vocab_order: The sequence in which vocabulary splits are correctly ordered.
        pipeline_num_virtual_layers_pre: The number of 'virtual' layers representing the
            computational cost of modules on the *first* stage, before the main
            layers (e.g., token embeddings and the vision encoder).
        pipeline_num_virtual_layers_post: The number of 'virtual' layers representing the
            computational cost of modules on the *last* stage, after the main
            layers (e.g., the final layer normalization and LM head).
    """

    layer: Qwen3p5MoELayerParameters

    num_hidden_layers: int
    full_attention_interval: int = Field(gt=0)
    rope_base: int
    mrope_section: tuple[int, int, int]
    max_position_ids: int

    split_vocab_size: dict[str, int]
    split_vocab_order: list[str]

    pipeline_num_virtual_layers_pre: int = 0
    pipeline_num_virtual_layers_post: int = 0

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        """Determines whether the given layer uses full attention.

        Args:
            layer_idx: The 0-based global layer index.

        Returns:
            True when the layer uses full attention, False for linear attention.
        """
        return (layer_idx + 1) % self.full_attention_interval == 0
