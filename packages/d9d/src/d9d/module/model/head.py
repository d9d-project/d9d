from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from d9d.module.block.head import ClassificationHead, EmbeddingHead, SplitLanguageModellingHead, TaskHead
from d9d.module.model.backbone import DecoderBackbone


class CausalLMHeadConfig(BaseModel):
    """Configuration for a causal language modeling head.

    The head takes its hidden size and split-vocabulary layout from the backbone, so this config has
    no fields besides its discriminator.

    Attributes:
        kind: Discriminator field. Always ``"causal_lm"``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["causal_lm"] = "causal_lm"


class ClassificationHeadConfig(BaseModel):
    """Configuration for a sequence/token classification head.

    Attributes:
        kind: Discriminator field. Always ``"classification"``.
        num_labels: The number of output classes.
        dropout: The dropout probability applied before the projection.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["classification"] = "classification"
    num_labels: int
    dropout: float = 0.0


class EmbeddingHeadConfig(BaseModel):
    """Configuration for a dense embedding head.

    Attributes:
        kind: Discriminator field. Always ``"embedding"``.
        embedding_dim: Size of the output embedding. ``None`` for no extra projection.
        normalize: Whether to apply L2 normalization to the final embeddings.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["embedding"] = "embedding"
    embedding_dim: int | None = None
    normalize: bool = False


AnyHeadConfig = Annotated[
    CausalLMHeadConfig | ClassificationHeadConfig | EmbeddingHeadConfig,
    Field(discriminator="kind"),
]
"""Closed, discriminated union of the built-in head configurations."""


def build_decoder_head(config: AnyHeadConfig, *, backbone: DecoderBackbone) -> TaskHead:
    """Builds a task head from its configuration and the decoder backbone it attaches to.

    The head takes ``hidden_size`` and the split-vocabulary layout from the backbone, so a config
    holds only task-specific fields. A custom head does not go through this factory: pass its
    ``TaskHead`` instance to the decoder directly.

    Args:
        config: Task head configuration selecting the head type and its task-specific fields.
        backbone: The decoder backbone the head attaches to. It provides the shared dimensions.

    Returns:
        An instantiated task head.

    Raises:
        ValueError: If the head configuration type is unknown.
    """
    match config:
        case CausalLMHeadConfig():
            return SplitLanguageModellingHead(
                split_vocab_size=backbone.split_vocab_size,
                split_order=backbone.split_vocab_order,
                hidden_size=backbone.hidden_size,
            )
        case ClassificationHeadConfig():
            return ClassificationHead(
                hidden_size=backbone.hidden_size,
                num_labels=config.num_labels,
                dropout=config.dropout,
            )
        case EmbeddingHeadConfig():
            return EmbeddingHead(
                hidden_size=backbone.hidden_size,
                embedding_dim=config.embedding_dim,
                normalize=config.normalize,
            )
        case _:
            raise ValueError(f"Unknown head config type ({type(config)}).")
