import dataclasses
from collections.abc import Mapping
from typing import Any, Generic, TypeAlias, TypeVar

import torch

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
    """

    position_ids: torch.Tensor
    hidden_states_agg_mask: torch.Tensor | None = None


@dataclasses.dataclass
class SequenceCausalLMHeadShared:
    """The shared input a causal language modeling head consumes.

    Attributes:
        labels: Target tokens for the loss computation.
    """

    labels: torch.Tensor


@dataclasses.dataclass
class SequencePoolingHeadShared:
    """The shared input a pooled head (classification/embedding) consumes.

    Attributes:
        pooling_mask: Binary mask indicating which token(s) to pool. You can use
            ``d9d.dataset.token_pooling_mask_from_attention_mask`` to build it from an attention mask.
    """

    pooling_mask: torch.Tensor | None = None


@dataclasses.dataclass
class SequenceHeadsShared:
    """The shared input a backbone composed with named task heads consumes on every stage.

    Attributes:
        sequence: The backbone shared input.
        heads: Each head's own shared input, keyed by the names the heads were composed under
            (read on the last stage).
    """

    sequence: SequenceShared
    heads: Mapping[str, Any]


SequenceHeadsOutput: TypeAlias = Mapping[str, Any]
"""
The output of a backbone composed with named task heads: each head's output, keyed by head name.

The key is the name the head was composed under, so two heads of the same type simply take
different keys and cannot collide.
"""


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
