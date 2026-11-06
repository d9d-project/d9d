import copy

import pytest
import torch
from d9d.models.decoder import DecoderWithHead, DecoderWithHeads
from d9d.models.io import SequenceHeadShared, SequenceHeadsShared, SequenceInput, SequenceTransfer
from d9d.models.segments import SequenceSegmentConstants, build_sequence_segments
from d9d.nn.hidden_states_aggregator.mean import HiddenStatesAggregatorMean
from d9d.pipelining.api import PipelineStageInfo, StageSegment, run_segments
from torch import nn


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states, position_embeddings):
        cos, sin = position_embeddings
        return torch.tanh(self.linear(hidden_states) + cos) + sin


def _direct_forward(modules, inputs, positions, first_stage, last_stage):
    if first_stage:
        hidden_states = modules["embed"](inputs.input_ids)
        snapshots = [hidden_states.mean(dim=1)]
    else:
        hidden_states = inputs.hidden_states
        snapshots = list(inputs.hidden_states_snapshot.unbind())
    for name in ("first", "second"):
        hidden_states = modules[name](hidden_states, positions)
        snapshots.append(hidden_states.mean(dim=1))
    if last_stage:
        hidden_states = modules["norm"](hidden_states)
    return SequenceTransfer(hidden_states, torch.stack(snapshots))


@pytest.mark.local
@pytest.mark.parametrize("first_stage", [False, True])
@pytest.mark.parametrize("last_stage", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("tied_layers", [False, True])
def test_sequence_segments_match_direct_forward_and_gradients(first_stage, last_stage, checkpointing, tied_layers):
    modules = nn.ModuleDict({"first": _Layer(), "second": _Layer()})
    if tied_layers:
        modules["second"] = modules["first"]
    if first_stage:
        modules["embed"] = nn.Embedding(8, 4)
    if last_stage:
        modules["norm"] = nn.LayerNorm(4)
    reference = copy.deepcopy(modules)
    positions = (torch.randn(2, 3, 4), torch.randn(2, 3, 4))
    constants = SequenceSegmentConstants(positions, HiddenStatesAggregatorMean(torch.ones(2, 3)))
    if first_stage:
        inputs = SequenceInput(torch.randint(0, 8, (2, 3)))
        reference_inputs = inputs
    else:
        inputs = SequenceTransfer(torch.randn(2, 3, 4, requires_grad=True), torch.randn(2, 2, 4, requires_grad=True))
        reference_inputs = SequenceTransfer(
            inputs.hidden_states.detach().clone().requires_grad_(),
            inputs.hidden_states_snapshot.detach().clone().requires_grad_(),
        )
    segments = build_sequence_segments(
        embed_tokens=modules["embed"] if first_stage else None,
        layers=[("first", modules["first"]), ("second", modules["second"])],
        norm=modules["norm"] if last_stage else None,
        enable_checkpointing=checkpointing,
    )

    expected_names = ["embed" if first_stage else "enter", "layer.first", "layer.second"]
    if last_stage:
        expected_names.append("norm")
    expected_names.append("pack")
    assert [segment.name for segment in segments] == expected_names
    for segment, name in zip(segments[1:3], ("first", "second"), strict=True):
        assert tuple(map(id, segment.module.parameters())) == tuple(map(id, modules[name].parameters()))

    actual = inputs
    for segment in segments:
        actual = segment.module(actual, constants)

    expected = _direct_forward(reference, reference_inputs, positions, first_stage, last_stage)

    torch.testing.assert_close(actual.hidden_states, expected.hidden_states)
    torch.testing.assert_close(actual.hidden_states_snapshot, expected.hidden_states_snapshot)
    (actual.hidden_states.square().sum() + actual.hidden_states_snapshot.square().sum()).backward()
    (expected.hidden_states.square().sum() + expected.hidden_states_snapshot.square().sum()).backward()
    for name, param in modules.named_parameters():
        torch.testing.assert_close(param.grad, reference.get_parameter(name).grad)
    if not first_stage:
        torch.testing.assert_close(inputs.hidden_states.grad, reference_inputs.hidden_states.grad)
        torch.testing.assert_close(inputs.hidden_states_snapshot.grad, reference_inputs.hidden_states_snapshot.grad)


class _ScaleSegment(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, activations, constants):
        return SequenceTransfer(activations * self.scale, None)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, inputs, shared):
        return run_segments(self, inputs, shared)

    def prepare_segment_constants(self, shared):
        return shared

    def stage_segments(self):
        return [StageSegment("scale", _ScaleSegment(self.scale))]


class _Head(nn.Module):
    def forward(self, hidden_states, shared):
        return hidden_states


@pytest.mark.local
def test_decoder_forward_calls_the_backbone_module():
    backbone = _Backbone()

    def add_ten(_module, _args, output):
        return SequenceTransfer(output.hidden_states + 10, None)

    backbone.register_forward_hook(add_ten)
    decoder = DecoderWithHead(backbone, _Head(), PipelineStageInfo(current_stage=0, num_stages=1))

    output = decoder(torch.tensor([[1.0, 2.0]]), SequenceHeadShared(sequence=None, head=None))

    torch.testing.assert_close(output, torch.tensor([[11.0, 12.0]]))


class _PlainBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.full((), 2.0))

    def forward(self, inputs, shared):
        return SequenceTransfer(inputs * self.scale, None)


class _ScaleHead(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = nn.Parameter(torch.full((), factor))

    def forward(self, hidden_states, shared):
        return hidden_states * self.factor + shared


@pytest.mark.local
@pytest.mark.parametrize("make_backbone", [_Backbone, _PlainBackbone], ids=["segments", "plain"])
@pytest.mark.parametrize("multi_head", [False, True])
@pytest.mark.parametrize("tied_weights", [False, True])
def test_decoder_segments_match_its_forward(make_backbone, multi_head, tied_weights):
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    backbone = make_backbone()
    heads = {"a": _ScaleHead(3.0), "b": _ScaleHead(5.0)} if multi_head else {"a": _ScaleHead(3.0)}
    if tied_weights:
        for head in heads.values():
            head.factor = backbone.scale
    if multi_head:
        decoder = DecoderWithHeads(backbone, heads, stage)
        shared = SequenceHeadsShared(sequence=None, heads={"a": 1.0, "b": 2.0})
    else:
        decoder = DecoderWithHead(backbone, heads["a"], stage)
        shared = SequenceHeadShared(sequence=None, head=1.0)
    inputs = torch.tensor([[1.0, 2.0]])
    parameter_ids = {name: id(p) for name, p in decoder.named_parameters()}
    state_keys = tuple(decoder.state_dict())

    expected = decoder(inputs, shared)
    actual = run_segments(decoder, inputs, shared)

    if multi_head:
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            torch.testing.assert_close(actual[name], value)
    else:
        torch.testing.assert_close(actual, expected)
    actual_loss = sum(value.sum() for value in actual.values()) if multi_head else actual.sum()
    expected_loss = sum(value.sum() for value in expected.values()) if multi_head else expected.sum()
    actual_grads = torch.autograd.grad(actual_loss, tuple(decoder.parameters()))
    expected_grads = torch.autograd.grad(expected_loss, tuple(decoder.parameters()))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    segment_parameters = {id(p) for segment in decoder.stage_segments() for p in segment.module.parameters()}
    assert segment_parameters == {id(p) for p in decoder.parameters()}
    assert {name: id(p) for name, p in decoder.named_parameters()} == parameter_ids
    assert tuple(decoder.state_dict()) == state_keys
