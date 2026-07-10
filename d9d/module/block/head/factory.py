from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field

from d9d.module.block.head.base import TaskHead
from d9d.module.block.head.classification import ClassificationHead
from d9d.module.block.head.embedding import EmbeddingHead
from d9d.module.block.head.language_modelling import SplitLanguageModellingHead

if TYPE_CHECKING:
    from d9d.module.model.decoder import DecoderBackbone
    from d9d.pipelining.api import PipelineStageInfo


class CausalLMHeadConfig(BaseModel):
    """Configuration for a causal language modeling head.

    The split-vocabulary layout and hidden size are derived from the backbone, so this
    config carries no task-specific fields beyond its discriminator.

    Attributes:
        kind: Discriminator field. Always "causal_lm".
    """

    kind: Literal["causal_lm"] = "causal_lm"


class ClassificationHeadConfig(BaseModel):
    """Configuration for a sequence/token classification head.

    Attributes:
        kind: Discriminator field. Always "classification".
        num_labels: The number of output classes.
        dropout: The dropout probability applied before the projection.
    """

    kind: Literal["classification"] = "classification"
    num_labels: int
    dropout: float = 0.0


class EmbeddingHeadConfig(BaseModel):
    """Configuration for a dense embedding head.

    Attributes:
        kind: Discriminator field. Always "embedding".
        embedding_dim: Dimensionality of the output embedding. None for no extra projection.
        normalize: Whether to apply L2 normalization to the final embeddings.
    """

    kind: Literal["embedding"] = "embedding"
    embedding_dim: int | None = None
    normalize: bool = False


AnyHeadConfig = Annotated[
    CausalLMHeadConfig | ClassificationHeadConfig | EmbeddingHeadConfig,
    Field(discriminator="kind"),
]
"""Closed, discriminated union of the built-in head configurations."""


def build_head(config: AnyHeadConfig, *, backbone: DecoderBackbone, stage: PipelineStageInfo) -> TaskHead:
    """Builds a task head from its configuration and the backbone it attaches to.

    Backbone-shared dimensions (``hidden_size``, the LM split-vocab layout) are derived from the
    backbone. A bespoke head a user writes for their own model is a :class:`TaskHead` instance
    passed directly to the decoder, bypassing this union entirely.

    Args:
        config: Task head configuration selecting the head type and its task-specific fields.
        backbone: The decoder backbone the head attaches to; provides shared dimensions.
        stage: Pipeline stage information for the head's decoder.

    Returns:
        An instantiated task head.

    Raises:
        ValueError: If an unknown head configuration type is provided.
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
            raise ValueError(f"Unknown head config type: {type(config)}")
