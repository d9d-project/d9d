import dataclasses
from collections.abc import Mapping
from typing import Generic, TypeAlias, TypeVar

import torch

TLeaf = TypeVar("TLeaf")
THeadShared = TypeVar("THeadShared")
THeadOutput = TypeVar("THeadOutput")


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
class SequenceHeadShared(Generic[THeadShared]):
    """The shared input a backbone composed with exactly one task head consumes on every stage.

    The single-head counterpart of :class:`SequenceHeadsShared`: the head is reached by field, not
    by name, and the model's output is that head's output unwrapped.

    Type parameters:
        THeadShared: The shared input accepted by the composed head.

    Attributes:
        sequence: The backbone shared input.
        head: The head's own shared input (read on the last stage).
    """

    sequence: SequenceShared
    head: THeadShared


@dataclasses.dataclass
class SequenceHeadsShared(Generic[THeadShared]):
    """The shared input a backbone composed with named task heads consumes on every stage.

    The parameter is the head shared input the composed heads accept: a single type for a model
    with one kind of head (e.g. ``SequenceHeadsShared[SequenceCausalLMHeadShared]``), or their
    union for a model composed of several kinds.

    Type parameters:
        THeadShared: The shared input accepted by the composed heads.

    Attributes:
        sequence: The backbone shared input.
        heads: Each head's own shared input, keyed by the names the heads were composed under
            (read on the last stage).
    """

    sequence: SequenceShared
    heads: Mapping[str, THeadShared]


SequenceHeadsOutput: TypeAlias = Mapping[str, THeadOutput]
"""
The output of a backbone composed with named task heads: each head's output, keyed by head name.

The key is the name the head was composed under, so two heads of the same type simply take
different keys and cannot collide. The parameter is the output the composed heads produce: a single
type for a model with one kind of head (e.g. ``SequenceHeadsOutput[SequenceCausalLMOutput]``), or
their union for a model composed of several kinds.
"""
