import copy

import pytest
import torch
from d9d.models.io import SequenceInput, SequenceTransfer
from d9d.models.segments import SequenceSegmentConstants, build_sequence_segments
from d9d.nn.hidden_states_aggregator.mean import HiddenStatesAggregatorMean
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
def test_sequence_segments_match_direct_forward_and_gradients(first_stage, last_stage, checkpointing):
    modules = nn.ModuleDict({"first": _Layer(), "second": _Layer()})
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
    assert segments[1].modules == (modules["first"],)
    assert segments[2].modules == (modules["second"],)

    actual = inputs
    for segment in segments:
        actual = segment.call(actual, constants)

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
