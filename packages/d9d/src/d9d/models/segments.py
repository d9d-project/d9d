import dataclasses
from collections.abc import Callable, Sequence
from functools import partial
from typing import cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from d9d.nn.hidden_states_aggregator import BaseHiddenStatesAggregator
from d9d.pipelining.api import StageSegment

from .io import SequenceInput, SequenceTransfer

_DecoderLayer = Callable[[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]


@dataclasses.dataclass
class SequenceSegmentConstants:
    """The non-differentiable values shared by the segments of a sequence transformer.

    Attributes:
        position_embeddings: The RoPE embeddings as a ``(cos, sin)`` tuple.
        aggregator: The aggregator of the hidden states snapshot.
    """

    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    aggregator: BaseHiddenStatesAggregator


@dataclasses.dataclass
class SequenceSegmentActivations:
    """The differentiable state that passes between the segments of a sequence transformer.

    Attributes:
        hidden_states: The output of the last segment. Shape: ``(batch, seq_len, hidden_size)``.
        snapshot: The snapshot received from the previous stage, or ``None``.
        collected: The values of the aggregator collected on this stage.
    """

    hidden_states: torch.Tensor
    snapshot: torch.Tensor | None
    collected: tuple[torch.Tensor, ...]


def _embed_tokens(
    inputs: SequenceInput, constants: SequenceSegmentConstants, *, embedding: nn.Module
) -> SequenceSegmentActivations:
    hidden_states = embedding(inputs.input_ids)
    return SequenceSegmentActivations(hidden_states, None, constants.aggregator.collect(hidden_states))


def _enter_transfer(
    inputs: SequenceTransfer[torch.Tensor], constants: SequenceSegmentConstants
) -> SequenceSegmentActivations:
    return SequenceSegmentActivations(inputs.hidden_states, inputs.hidden_states_snapshot, ())


def _run_decoder_layer(
    activations: SequenceSegmentActivations,
    constants: SequenceSegmentConstants,
    *,
    layer: _DecoderLayer,
    enable_checkpointing: bool,
) -> SequenceSegmentActivations:
    if enable_checkpointing:
        hidden_states = cast(
            torch.Tensor,
            checkpoint(layer, activations.hidden_states, constants.position_embeddings, use_reentrant=False),
        )
    else:
        hidden_states = layer(activations.hidden_states, constants.position_embeddings)
    collected = activations.collected + constants.aggregator.collect(hidden_states)
    return SequenceSegmentActivations(hidden_states, activations.snapshot, collected)


def _normalize(
    activations: SequenceSegmentActivations, constants: SequenceSegmentConstants, *, norm: nn.Module
) -> SequenceSegmentActivations:
    return SequenceSegmentActivations(norm(activations.hidden_states), activations.snapshot, activations.collected)


def _pack_transfer(
    activations: SequenceSegmentActivations, constants: SequenceSegmentConstants
) -> SequenceTransfer[torch.Tensor]:
    return SequenceTransfer(
        hidden_states=activations.hidden_states,
        hidden_states_snapshot=constants.aggregator.pack(activations.collected, activations.snapshot),
    )


def build_sequence_segments(
    embed_tokens: nn.Module | None,
    layers: Sequence[tuple[str, nn.Module]],
    norm: nn.Module | None,
    enable_checkpointing: bool,
) -> list[StageSegment]:
    """Builds the segments of a sequence transformer stage.

    Args:
        embed_tokens: The token embeddings on the first stage, ``None`` otherwise.
        layers: The names and modules of the decoder layers of this stage, in order.
        norm: The final norm on the last stage, ``None`` otherwise.
        enable_checkpointing: Whether to apply activation checkpointing to each decoder layer.

    Returns:
        The segments, in forward order.
    """
    if embed_tokens is not None:
        segments = [StageSegment("embed", (embed_tokens,), partial(_embed_tokens, embedding=embed_tokens))]
    else:
        segments = [StageSegment("enter", (), _enter_transfer)]
    segments.extend(
        StageSegment(
            f"layer.{name}",
            (layer,),
            partial(_run_decoder_layer, layer=layer, enable_checkpointing=enable_checkpointing),
        )
        for name, layer in layers
    )
    if norm is not None:
        segments.append(StageSegment("norm", (norm,), partial(_normalize, norm=norm)))
    segments.append(StageSegment("pack", (), _pack_transfer))
    return segments
