from pydantic import BaseModel, ConfigDict


class Qwen3DenseLayerParameters(BaseModel):
    """Configuration for a single Qwen3 Dense layer.

    Attributes:
        hidden_size: Size of the hidden states.
        intermediate_size: Size of the feed-forward hidden layer.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key and value attention heads.
        rms_norm_eps: Epsilon of the RMSNorm layers.
        head_dim: Size of a single attention head.
    """

    model_config = ConfigDict(extra="forbid")

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    head_dim: int


class Qwen3DenseParameters(BaseModel):
    """Configuration for the Qwen3 Dense model backbone.

    Attributes:
        layer: Configuration shared across all transformer layers.
        num_hidden_layers: Total number of transformer layers.
        rope_base: Base of the RoPE frequencies.
        max_position_ids: Maximum sequence length.
        split_vocab_size: Mapping of vocabulary segment names to their sizes.
        split_vocab_order: Order in which the vocabulary segments are concatenated.
        pipeline_num_virtual_layers_pre: Number of virtual layers that stand for the compute cost of
            modules on the first stage before the main layers, such as token embeddings. Pipeline
            layer distribution uses it.
        pipeline_num_virtual_layers_post: Number of virtual layers that stand for the compute cost of
            modules on the last stage after the main layers, such as the final norm and the LM head.
            Pipeline layer distribution uses it.
    """

    model_config = ConfigDict(extra="forbid")

    layer: Qwen3DenseLayerParameters

    num_hidden_layers: int
    rope_base: int
    max_position_ids: int

    split_vocab_size: dict[str, int]
    split_vocab_order: list[str]

    pipeline_num_virtual_layers_pre: int = 0
    pipeline_num_virtual_layers_post: int = 0
