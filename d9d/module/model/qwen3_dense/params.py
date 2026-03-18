from pydantic import BaseModel


class Qwen3DenseLayerParameters(BaseModel):
    """
    Configuration parameters for a single Qwen3 Dense layer.

    Attributes:
        hidden_size: Dimension of the model's hidden states.
        intermediate_size: Dimension of the feed-forward hidden state.
        num_attention_heads: Number of attention heads for the query.
        num_key_value_heads: Number of attention heads for key and value.
        rms_norm_eps: Epsilon value found in the RMSNorm layers.
        head_dim: Dimension of a single attention head.
    """

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    head_dim: int


class Qwen3DenseParameters(BaseModel):
    """
    Configuration parameters for the Qwen3 Dense model backbone.

    Attributes:
        layer: Configuration shared across all transformer layers.
        num_hidden_layers: The total number of transformer layers.
        rope_base: Base value for RoPE frequency calculation.
        max_position_ids: Maximum sequence length.
        split_vocab_size: A dictionary mapping vocabulary segment names to their sizes.
        split_vocab_order: The sequence in which vocabulary splits are correctly ordered.
        pipeline_num_virtual_layers_pre: The number of 'virtual' layers representing the
            computational cost of modules on the *first* stage, before the main
            layers (e.g., token and positional embeddings).
        pipeline_num_virtual_layers_post: The number of 'virtual' layers representing the
            computational cost of modules on the *last* stage, after the main
            layers (e.g., the final layer normalization and LM head).
    """

    layer: Qwen3DenseLayerParameters

    num_hidden_layers: int
    rope_base: int
    max_position_ids: int

    split_vocab_size: dict[str, int]
    split_vocab_order: list[str]

    pipeline_num_virtual_layers_pre: int = 0
    pipeline_num_virtual_layers_post: int = 0


class Qwen3DenseForCausalLMParameters(BaseModel):
    """
    Configuration parameters for Qwen3 Dense model with a Causal Language Modeling head.

    Attributes:
        model: The configuration for the underlying Qwen3 Dense model.
    """

    model: Qwen3DenseParameters


class Qwen3DenseForClassificationParameters(BaseModel):
    """
    Configuration parameters for Qwen3 Dense model with a token/sequence classification head.

    Attributes:
        model: The configuration for the underlying Qwen3 Dense model.
        num_labels: The number of output labels for classification.
        classifier_dropout: The dropout probability for the classification head.
    """

    model: Qwen3DenseParameters
    num_labels: int
    classifier_dropout: float
