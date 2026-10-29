import dataclasses
from collections.abc import Mapping
from typing import Generic, TypeAlias, TypeVar

import torch

TLeaf = TypeVar("TLeaf")
THeadShared = TypeVar("THeadShared")
THeadOutput = TypeVar("THeadOutput")


@dataclasses.dataclass
class SequenceInput:
    """The input of a sequence transformer: the token ids fed to the first stage.

    Attributes:
        input_ids: Indices of the input tokens. Shape: ``(batch, seq_len)``.
    """

    input_ids: torch.Tensor


@dataclasses.dataclass
class SequenceTransfer(Generic[TLeaf]):
    """The object moved between adjacent stages of a sequence transformer.

    Attributes:
        hidden_states: The output of the last layer of the sending stage.
            Shape: ``(batch, seq_len, hidden_size)``.
        hidden_states_snapshot: The aggregated hidden states of the embeddings and of all layers so far, or
            ``None`` if snapshotting is disabled. Shape: ``(num_layers, batch, hidden_size)``.
    """

    hidden_states: TLeaf
    hidden_states_snapshot: TLeaf | None = None


@dataclasses.dataclass
class SequenceShared:
    """The shared input the transformer backbone consumes on every stage.

    Attributes:
        position_ids: The position of each token in the position embeddings.
            Shape: ``(batch, seq_len)``.
        hidden_states_agg_mask: Mask of the tokens to aggregate into hidden state snapshots, or ``None``
            if snapshotting is disabled. Shape: ``(batch, seq_len)``.
    """

    position_ids: torch.Tensor
    hidden_states_agg_mask: torch.Tensor | None = None


@dataclasses.dataclass
class SequenceHeadShared(Generic[THeadShared]):
    """The shared input a backbone composed with exactly one task head consumes on every stage.

    This is the single-head counterpart of ``SequenceHeadsShared``. The head input is a field, not a
    named entry, and the model returns the head's output unwrapped.

    Type parameters:
        THeadShared: The shared input accepted by the composed head.

    Attributes:
        sequence: The backbone shared input.
        head: The head's own shared input. Read on the last stage only.
    """

    sequence: SequenceShared
    head: THeadShared


@dataclasses.dataclass
class SequenceHeadsShared(Generic[THeadShared]):
    """The shared input a backbone composed with named task heads consumes on every stage.

    The type parameter is the shared input of the composed heads. It is a single type for heads of
    one kind (e.g. ``SequenceHeadsShared[SequenceCausalLMHeadShared]``), or a union for heads of
    several kinds.

    Type parameters:
        THeadShared: The shared input accepted by the composed heads.

    Attributes:
        sequence: The backbone shared input.
        heads: Each head's own shared input, keyed by head name. Read on the last stage only.
    """

    sequence: SequenceShared
    heads: Mapping[str, THeadShared]


SequenceHeadsOutput: TypeAlias = Mapping[str, THeadOutput]
"""
The output of a backbone composed with named task heads: each head's output, keyed by head name.

Two heads of the same type have different names, so their outputs cannot collide. The type
parameter is the output of the composed heads. It is a single type for heads of one kind (e.g.
``SequenceHeadsOutput[SequenceCausalLMOutput]``), or a union for heads of several kinds.
"""
