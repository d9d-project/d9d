import dataclasses
from typing import Generic, TypeVar

import torch

from d9d.module.block.attention import SequencePacking

TLeaf = TypeVar("TLeaf")


@dataclasses.dataclass
class SequenceInput:
    """The inputs for a sequence transformer: the token ids fed to the first stage.

    Attributes:
        input_ids: Indices of input sequence tokens, shape ``[batch, seq]``.
    """

    input_ids: torch.Tensor


@dataclasses.dataclass
class SequenceTransfer(Generic[TLeaf]):
    """The object moved between adjacent stages of a sequence transformer.

    Attributes:
        hidden_states: The output of the last layer of the sending stage, shape ``[batch, seq, hidden]``.
        hidden_states_snapshot: The accumulated aggregated hidden states carried across stages when
            snapshotting is enabled, else ``None``.
    """

    hidden_states: TLeaf
    hidden_states_snapshot: TLeaf | None = None


@dataclasses.dataclass
class SequenceShared:
    """The shared input the transformer backbone consumes on every stage.

    Attributes:
        position_ids: Indices of positions of each token in the position embeddings.
        hidden_states_agg_mask: Mask used to aggregate hidden states for snapshots, if enabled.
        packing: When set, the batch is a single
            packed row and attention is computed block-diagonally over its segments; ``None`` gives
            the dense causal path.
    """

    position_ids: torch.Tensor
    hidden_states_agg_mask: torch.Tensor | None = None
    packing: SequencePacking | None = None


@dataclasses.dataclass
class SequenceCausalLMShared:
    """The shared input for causal language modeling, broadcast to every stage.

    Attributes:
        sequence: The backbone shared input.
        labels: Target tokens for the loss computation (used on the last stage).
    """

    sequence: SequenceShared
    labels: torch.Tensor | None = None


@dataclasses.dataclass
class SequencePoolingShared:
    """The shared input for pooled heads (classification/embedding), broadcast to every stage.

    Attributes:
        sequence: The backbone shared input.
        pooling_mask: Binary mask indicating which token(s) to pool (used on the last stage). You can
            use ``d9d.dataset.token_pooling_mask_from_attention_mask`` to build it from an attention mask.
    """

    sequence: SequenceShared
    pooling_mask: torch.Tensor | None = None


@dataclasses.dataclass
class SequenceCausalLMOutput:
    """The output of a causal language modeling head.

    Attributes:
        logps: Per-token log-probabilities / loss, shape ``[batch, seq]``.
    """

    logps: torch.Tensor


@dataclasses.dataclass
class SequenceClassificationOutput:
    """The output of a classification head.

    Attributes:
        scores: Classification logits, shape ``[num_pooled_tokens, num_labels]`` when a pooling mask
            selects tokens, or ``[batch, seq, num_labels]`` when no pooling mask is used.
    """

    scores: torch.Tensor


@dataclasses.dataclass
class SequenceEmbeddingOutput:
    """The output of an embedding head.

    Attributes:
        embeddings: Pooled embeddings, shape ``[num_pooled_tokens, embedding_dim]`` when a pooling mask
            selects tokens, or ``[batch, seq, embedding_dim]`` when no pooling mask is used.
    """

    embeddings: torch.Tensor
