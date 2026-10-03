import dataclasses
from collections.abc import Mapping, Sequence

import pytest
import torch
from d9d.core.types import TensorSpec
from d9d.module.base import MediaSegments
from d9d.module.model import MultimodalBackbone
from d9d.module.model.io import MultimodalSequenceInput, SequenceInput, SequenceShared, SequenceTransfer
from d9d.pipelining.api import PipelineStageInfo, StageBoundary
from torch import nn

_HIDDEN = 4
_VOCAB = 10


class _FakeEmbeddings(nn.Module):
    """A token embedding table mapping every id to a constant row, so merges are easy to assert."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(_VOCAB, _HIDDEN))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[input_ids]


class _FakeBackbone(nn.Module):
    """Records what it was handed, and returns the embeddings it consumed as hidden states."""

    def __init__(self, stage: PipelineStageInfo):
        super().__init__()
        self._stage = stage
        self.seen_inputs: SequenceInput | SequenceTransfer[torch.Tensor] | None = None
        if stage.is_current_stage_first:
            self.embed_tokens = _FakeEmbeddings()
        self.reset_called = False

    @property
    def hidden_size(self) -> int:
        return _HIDDEN

    @property
    def split_vocab_size(self) -> Mapping[str, int]:
        return {"a": _VOCAB}

    @property
    def split_vocab_order(self) -> Sequence[str]:
        return ["a"]

    def forward(self, inputs, shared: SequenceShared) -> SequenceTransfer[torch.Tensor]:
        self.seen_inputs = inputs
        if isinstance(inputs, SequenceInput):
            hidden = inputs.inputs_embeds if inputs.inputs_embeds is not None else self.embed_tokens(inputs.input_ids)
        else:
            hidden = inputs.hidden_states
        return SequenceTransfer(hidden_states=hidden)

    def reset_parameters(self) -> None:
        self.reset_called = True

    def stage_transfer_spec(self, pipeline_input, boundary: StageBoundary) -> SequenceTransfer[TensorSpec]:
        batch, seq = pipeline_input.input_ids.shape
        return SequenceTransfer(hidden_states=TensorSpec(shape=(batch, seq, _HIDDEN), dtype=torch.float32))


class _FakeEncoder(nn.Module):
    """Emits one distinctive embedding row per media token."""

    def __init__(self, num_tokens: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self._num_tokens = num_tokens
        self.call_count = 0
        self.reset_called = False

    def forward(self, media: MediaSegments) -> torch.Tensor:
        self.call_count += 1
        rows = torch.arange(1, self._num_tokens + 1, dtype=torch.float32).unsqueeze(-1)
        return rows.expand(self._num_tokens, _HIDDEN) * self.scale

    def reset_parameters(self) -> None:
        self.reset_called = True


def _media(num_tokens: int) -> MediaSegments:
    return MediaSegments(
        features=torch.zeros(num_tokens, 3),
        grid_thw=torch.tensor([[1, 1, num_tokens]], dtype=torch.long),
    )


def _build(stage: PipelineStageInfo, num_media_tokens: int = 2):
    backbone = _FakeBackbone(stage)
    encoder = _FakeEncoder(num_media_tokens)
    return MultimodalBackbone(backbone, encoder, stage), backbone, encoder


@pytest.mark.local
def test_merges_media_into_placeholder_positions():
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    model, backbone, _ = _build(stage)

    inputs = MultimodalSequenceInput(
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        media=_media(2),
        media_token_mask=torch.tensor([[False, True, True, False]]),
    )

    out = model(inputs, SequenceShared(position_ids=torch.zeros(1, 4, dtype=torch.long)))

    # placeholder rows carry the encoder output, the rest keep the (zero) token embeddings
    torch.testing.assert_close(out.hidden_states[0, 1], torch.ones(_HIDDEN))
    torch.testing.assert_close(out.hidden_states[0, 2], torch.full((_HIDDEN,), 2.0))
    torch.testing.assert_close(out.hidden_states[0, 0], torch.zeros(_HIDDEN))
    torch.testing.assert_close(out.hidden_states[0, 3], torch.zeros(_HIDDEN))

    # the wrapped backbone was handed a plain SequenceInput carrying the merged embeddings
    assert isinstance(backbone.seen_inputs, SequenceInput)
    assert backbone.seen_inputs.inputs_embeds is not None


@pytest.mark.local
def test_reports_backbone_dimensions():
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    model, _, _ = _build(stage)

    assert model.hidden_size == _HIDDEN
    assert model.split_vocab_size == {"a": _VOCAB}
    assert list(model.split_vocab_order) == ["a"]


@pytest.mark.local
def test_encoder_only_on_first_stage_and_later_stages_pass_through():
    stage = PipelineStageInfo(current_stage=1, num_stages=2)
    model, backbone, encoder = _build(stage)

    assert not hasattr(model, "encoder"), "the encoder must not be attached off the first stage"

    transfer = SequenceTransfer(hidden_states=torch.randn(1, 4, _HIDDEN))
    out = model(transfer, SequenceShared(position_ids=torch.zeros(1, 4, dtype=torch.long)))

    torch.testing.assert_close(out.hidden_states, transfer.hidden_states)
    assert encoder.call_count == 0
    assert backbone.seen_inputs is transfer


@pytest.mark.local
def test_gradients_reach_the_encoder():
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    model, _, encoder = _build(stage, num_media_tokens=1)

    inputs = MultimodalSequenceInput(
        input_ids=torch.zeros(1, 3, dtype=torch.long),
        media=_media(1),
        media_token_mask=torch.tensor([[False, True, False]]),
    )

    out = model(inputs, SequenceShared(position_ids=torch.zeros(1, 3, dtype=torch.long)))
    out.hidden_states.sum().backward()

    assert encoder.scale.grad is not None
    assert encoder.scale.grad.abs().sum() > 0


@pytest.mark.local
def test_empty_media_still_runs_the_encoder():
    # The empty-media convention: a media-free microbatch keeps every collective aligned by
    # running the encoder on a dummy segment whose contribution is zeroed out.
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    model, _, encoder = _build(stage, num_media_tokens=1)

    inputs = MultimodalSequenceInput(
        input_ids=torch.zeros(1, 3, dtype=torch.long),
        media=_media(1),
        media_token_mask=torch.zeros(1, 3, dtype=torch.bool),
    )

    out = model(inputs, SequenceShared(position_ids=torch.zeros(1, 3, dtype=torch.long)))
    out.hidden_states.sum().backward()

    assert encoder.call_count == 1
    assert encoder.scale.grad is not None
    torch.testing.assert_close(encoder.scale.grad, torch.zeros(()))


@pytest.mark.local
def test_rejects_placeholder_count_mismatch():
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    model, _, _ = _build(stage, num_media_tokens=2)

    inputs = MultimodalSequenceInput(
        input_ids=torch.zeros(1, 4, dtype=torch.long),
        media=_media(2),
        media_token_mask=torch.tensor([[True, True, True, False]]),  # 3 placeholders vs 2 tokens
    )

    with pytest.raises(ValueError, match="does not match"):
        model(inputs, SequenceShared(position_ids=torch.zeros(1, 4, dtype=torch.long)))


@pytest.mark.local
def test_reset_parameters_reaches_backbone_and_encoder():
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    model, backbone, encoder = _build(stage)

    model.reset_parameters()

    assert backbone.reset_called
    assert encoder.reset_called


@pytest.mark.local
def test_stage_transfer_spec_delegates_to_backbone():
    stage = PipelineStageInfo(current_stage=0, num_stages=2)
    model, _, _ = _build(stage)

    inputs = MultimodalSequenceInput(
        input_ids=torch.zeros(2, 5, dtype=torch.long),
        media=_media(2),
        media_token_mask=torch.zeros(2, 5, dtype=torch.bool),
    )

    spec = model.stage_transfer_spec(inputs, StageBoundary.outgoing)

    assert spec.hidden_states.shape == (2, 5, _HIDDEN)


@pytest.mark.local
def test_is_a_decoder_backbone_by_structure():
    # The composition must satisfy the backbone surface a task head composition calls on it, so it
    # drops into DecoderForCausalLM exactly as a text-only backbone does.
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    model, _, _ = _build(stage)

    for attr in ("hidden_size", "split_vocab_size", "split_vocab_order", "reset_parameters", "stage_transfer_spec"):
        assert hasattr(model, attr), f"missing backbone surface: {attr}"


@pytest.mark.local
def test_sequence_input_defaults_to_embedding_lookup():
    # inputs_embeds is additive: omitting it keeps the pre-existing token-id path intact.
    inputs = SequenceInput(input_ids=torch.zeros(1, 2, dtype=torch.long))
    assert inputs.inputs_embeds is None
    assert [f.name for f in dataclasses.fields(SequenceInput)] == ["input_ids", "inputs_embeds"]
