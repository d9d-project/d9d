from weakref import ref

import pytest
import torch
from d9d.dist_context import BASE_DOMAIN, DATA_SHARD_DIMS, BaseDim, DeviceMeshParameters
from d9d.models import DecoderForCausalLM
from d9d.models.io import SequenceHeadShared, SequenceInput, SequenceShared, SequenceTransfer
from d9d.models.qwen3_dense import Qwen3DenseLayerParameters, Qwen3DenseModel, Qwen3DenseParameters
from d9d.models.qwen3_moe import Qwen3MoELayerParameters, Qwen3MoEModel, Qwen3MoEParameters
from d9d.nn.head import SequenceCausalLMHeadShared
from d9d.nn.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.parallel.runtime import ParameterRuntime
from d9d.parallel.runtime.plan import plan_stages
from d9d.pipelining.api import PipelineStageInfo, mark_window, run_steps
from d9d.pipelining.factory import (
    PipelineSchedule1F1BConfig,
    PipelineScheduleDualPipeVConfig,
    PipelineScheduleGPipeConfig,
    PipelineScheduleLoopedBFSConfig,
    PipelineScheduleZeroBubbleVConfig,
    build_program_builder,
    build_schedule,
)
from d9d.pipelining.stage.computations import (
    BackwardComputeHandler,
    BackwardSeedLoss,
    BackwardSeedTransfer,
    ForwardComputeHandler,
)
from torch import nn

from d9d_dev.nn.helper.tolerances import FP32_REORDERED_SUM_TOLERANCE
from d9d_dev.parallel.runtime_step import global_weight
from d9d_dev.pipelining.step_handler import FunctionStepHandler

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_COMMON = {"hidden_size": 64, "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-5, "head_dim": 16}
_TOP = {
    "num_hidden_layers": 6,
    "rope_base": 10000,
    "max_position_ids": 128,
    "split_vocab_size": {"a": 50},
    "split_vocab_order": ["a"],
}
_BATCH, _SEQ, _HIDDEN = 2, 8, 64
_TOLERANCE = FP32_REORDERED_SUM_TOLERANCE


def _build_backbone(kind, stage, snapshot, checkpointing, num_layers=6):
    top = {**_TOP, "num_hidden_layers": num_layers}
    mode = HiddenStatesAggregationMode.mean if snapshot else HiddenStatesAggregationMode.no
    if kind == "moe":
        layer = Qwen3MoELayerParameters(intermediate_size=32, num_experts=4, experts_top_k=2, **_COMMON)
        return Qwen3MoEModel(Qwen3MoEParameters(layer=layer, **top), stage, mode, checkpointing)
    layer = Qwen3DenseLayerParameters(intermediate_size=96, **_COMMON)
    return Qwen3DenseModel(Qwen3DenseParameters(layer=layer, **top), stage, mode, checkpointing)


def _set_weights(model):
    # The two runs of a comparison build their own models, which need equal weights.
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(global_weight(name, param))
    return model


def _materialize(model):
    model.to_empty(device=_DEVICE)
    with torch.no_grad():
        model.reset_parameters()
    return _set_weights(model)


def _middle_inputs(snapshot):
    hidden = torch.randn(_BATCH, _SEQ, _HIDDEN, device=_DEVICE)
    snap = torch.randn(3, _BATCH, _HIDDEN, device=_DEVICE) if snapshot else None
    return SequenceTransfer(hidden_states=hidden, hidden_states_snapshot=snap)


def _leaf(value):
    # Both runs of a comparison read the same inputs, each through its own leaf.
    return None if value is None else value.detach().clone().requires_grad_()


def _shared(snapshot):
    positions = torch.arange(_SEQ, device=_DEVICE).expand(_BATCH, _SEQ)
    mask = torch.ones(_BATCH, _SEQ, device=_DEVICE) if snapshot else None
    return SequenceShared(position_ids=positions, hidden_states_agg_mask=mask)


def _wave(like, phase):
    # A fixed gradient that differs per microbatch, so both runs of a comparison get the same one.
    return torch.linspace(-1, 1, like.numel(), device=like.device).add(phase).sin().reshape(like.shape)


def _cotangent(outputs, phase):
    snapshot = outputs.hidden_states_snapshot
    return SequenceTransfer(
        hidden_states=_wave(outputs.hidden_states, phase),
        hidden_states_snapshot=None if snapshot is None else _wave(snapshot, phase),
    )


def _ids():
    return torch.randint(0, 50, (_BATCH, _SEQ), device=_DEVICE)


def _runtime(stage_index, model):
    runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mib=32)
    runtime.register_stage(stage_index, model)
    runtime.bind()
    return runtime


def _run_middle_stage(kind, snapshot, checkpointing, split, windowed, data):
    """Runs two microbatches on a middle stage: F0, F1, then B1, B0 or I1, I0, W1, W0."""
    stage = PipelineStageInfo(current_stage=1, num_stages=3)
    model = _materialize(_build_backbone(kind, stage, snapshot, checkpointing))
    runtime = _runtime(1, model) if windowed else None
    forward = ForwardComputeHandler(1, model, runtime=runtime)
    backward = BackwardComputeHandler(1, model, has_input_peer=True, runtime=runtime)
    shared = _shared(snapshot)

    for mb in range(2):
        inputs = SequenceTransfer(_leaf(data[mb].hidden_states), _leaf(data[mb].hidden_states_snapshot))
        forward.run(mb, inputs, shared)

    input_grads = {}
    for mb in reversed(range(2)):
        cache = forward.pop_cache(mb)
        inputs, outputs, chain = cache.inputs, cache.outputs, cache.chain
        assert (chain is not None) == windowed
        seed = BackwardSeedTransfer(outputs=outputs, output_grads=_cotangent(outputs, mb))
        if split:
            backward.backward_input(mb, inputs, seed, chain=chain)
        else:
            backward.backward_full(mb, inputs, seed, chain=chain)
        input_grads[mb] = backward.pop_for_sending(mb)
    if split:
        for mb in reversed(range(2)):
            backward.backward_weight(mb)

    return {n: p.grad.clone() for n, p in model.named_parameters()}, input_grads


def _run_single_stage(kind, snapshot, checkpointing, split, windowed, data):
    """Runs two microbatches of the whole model with a head, from token ids to the loss."""
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    backbone = _build_backbone(kind, stage, snapshot, checkpointing)
    model = _materialize(DecoderForCausalLM(backbone, stage))
    runtime = _runtime(0, model) if windowed else None
    forward = ForwardComputeHandler(0, model, runtime=runtime)
    backward = BackwardComputeHandler(0, model, has_input_peer=False, runtime=runtime)

    for mb, ids in enumerate(data):
        shared = SequenceHeadShared(sequence=_shared(snapshot), head=SequenceCausalLMHeadShared(labels=ids))
        forward.run(mb, SequenceInput(input_ids=ids), shared)

    for mb in reversed(range(2)):
        cache = forward.pop_cache(mb)
        inputs, outputs, chain = cache.inputs, cache.outputs, cache.chain
        assert (chain is not None) == windowed
        seed = BackwardSeedLoss(loss=outputs.cross_entropy.sum())
        if split:
            backward.backward_input(mb, inputs, seed, chain=chain)
            backward.backward_weight(mb)
        else:
            backward.backward_full(mb, inputs, seed, chain=chain)

    return {n: p.grad.clone() for n, p in model.named_parameters()}


def _assert_close(left, right):
    assert left.keys() == right.keys()
    for name, value in left.items():
        torch.testing.assert_close(
            value,
            right[name],
            rtol=_TOLERANCE.rtol,
            atol=_TOLERANCE.atol,
            msg=lambda message, name=name: f"{name}: {message}",
        )


@pytest.mark.local
@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("snapshot", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_middle_stage_matches_one_graph(kind, snapshot, checkpointing, split):
    data = [_middle_inputs(snapshot) for _ in range(2)]
    whole, whole_inputs = _run_middle_stage(kind, snapshot, checkpointing, split, windowed=False, data=data)
    cut, cut_inputs = _run_middle_stage(kind, snapshot, checkpointing, split, windowed=True, data=data)

    _assert_close(cut, whole)
    for mb in range(2):
        torch.testing.assert_close(
            cut_inputs[mb].hidden_states, whole_inputs[mb].hidden_states, rtol=_TOLERANCE.rtol, atol=_TOLERANCE.atol
        )
        if snapshot:
            torch.testing.assert_close(
                cut_inputs[mb].hidden_states_snapshot,
                whole_inputs[mb].hidden_states_snapshot,
                rtol=_TOLERANCE.rtol,
                atol=_TOLERANCE.atol,
            )


@pytest.mark.local
@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_single_stage_with_head_matches_one_graph(kind, checkpointing, split):
    data = [_ids() for _ in range(2)]
    whole = _run_single_stage(kind, False, checkpointing, split, windowed=False, data=data)
    cut = _run_single_stage(kind, False, checkpointing, split, windowed=True, data=data)

    _assert_close(cut, whole)


class _FanOutFirst(nn.Linear):
    def forward(self, inputs, shared):
        return super().forward(inputs), super().forward(inputs) * 2


class _FanOutSecond(nn.Linear):
    def forward(self, inputs, shared):
        return inputs[0], inputs[1], super().forward(inputs[0])


class _FanOutThird(nn.Linear):
    def forward(self, inputs, shared):
        return super().forward(inputs[0]) + inputs[2]


class _FanOut(nn.Module):
    """One tensor has two consumers, one tensor passes through a region, one output stays unused."""

    def __init__(self):
        super().__init__()
        self.first = mark_window(_FanOutFirst(4, 4))
        self.second = mark_window(_FanOutSecond(4, 4))
        self.third = mark_window(_FanOutThird(4, 4))

    async def forward_steps(self, steps, inputs, shared):
        inputs = await steps.call(self.first, inputs, shared)
        inputs = await steps.call(self.second, inputs, shared)
        return await steps.call(self.third, inputs, shared)

    def forward(self, inputs, shared):
        return run_steps(self, inputs, shared)


@pytest.mark.local
def test_fan_out_pass_through_and_unused_output():
    data = torch.randn(2, 4)

    def run(windowed):
        model = _set_weights(_FanOut())
        runtime = _runtime(1, model) if windowed else None
        forward = ForwardComputeHandler(1, model, runtime=runtime)
        backward = BackwardComputeHandler(1, model, has_input_peer=True, runtime=runtime)
        forward.run(0, _leaf(data), None)
        cache = forward.pop_cache(0)
        stage_inputs, outputs, chain = cache.inputs, cache.outputs, cache.chain
        backward.backward_full(0, stage_inputs, BackwardSeedLoss(loss=outputs.sum()), chain=chain)
        return {n: p.grad.clone() for n, p in model.named_parameters()}, backward.pop_for_sending(0)

    whole, whole_input = run(False)
    cut, cut_input = run(True)

    _assert_close(cut, whole)
    torch.testing.assert_close(cut_input, whole_input, rtol=_TOLERANCE.rtol, atol=_TOLERANCE.atol)


@pytest.mark.local
def test_groups_cover_all_trainable_parameters():
    model = _FanOut()
    plan = plan_stages({0: model}, None, True, 32, frozenset())[0]
    assert {id(p.sharded) for p in plan.parameters} == {id(p) for p in model.parameters()}


def _run_pipeline(dist_ctx, schedule_config, kind, checkpointing, windowed, data):
    models = {}
    activations = []

    def _provider(stage):
        model = _materialize(
            DecoderForCausalLM(_build_backbone(kind, stage, False, checkpointing, num_layers=16), stage)
        )
        for layer in model.model.layers.values():
            layer.register_forward_hook(lambda _module, _args, output: activations.append(ref(output)))
        models[stage.current_stage] = model
        return model

    runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mib=32) if windowed else None
    schedule_info, _ = build_schedule(
        pp_mesh=dist_ctx.mesh_for(BASE_DOMAIN)[BaseDim.PP],
        program_builder=build_program_builder(schedule_config),
        model_provider=_provider,
        runtime=runtime,
    )
    if windowed:
        runtime.bind()

    inputs, shared = [], []
    for ids in data:
        inputs.append(SequenceInput(input_ids=ids))
        shared.append(SequenceHeadShared(sequence=_shared(False), head=SequenceCausalLMHeadShared(labels=ids)))

    schedule_info.schedule.step(
        inputs_microbatches=tuple(inputs),
        shared_microbatches=tuple(shared),
        handler=FunctionStepHandler(lambda result, _: result.cross_entropy.sum()),
    )

    # A step must not keep the activations of its microbatches alive after it ends.
    assert activations
    assert all(activation() is None for activation in activations)

    return {(i, n): p.grad.clone() for i, model in models.items() for n, p in model.named_parameters()}


@pytest.mark.distributed
@pytest.mark.parametrize(
    "schedule_config",
    [
        PipelineScheduleGPipeConfig(),
        PipelineScheduleLoopedBFSConfig(num_stages_per_rank=2),
        PipelineSchedule1F1BConfig(num_stages_per_rank=1, zero_bubble=False),
        PipelineSchedule1F1BConfig(num_stages_per_rank=2, zero_bubble=True),
        PipelineScheduleZeroBubbleVConfig(),
        PipelineScheduleDualPipeVConfig(),
    ],
)
@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_pipeline_schedules_match_one_graph(dist_ctx_factory, schedule_config, kind, checkpointing):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(pipeline_parallel=8))
    # Drawn before the models consume random numbers, so every rank, from the first to the last stage, gets the
    # same tokens.
    data = [_ids() for _ in range(16)]

    whole = _run_pipeline(dist_ctx, schedule_config, kind, checkpointing, windowed=False, data=data)
    cut = _run_pipeline(dist_ctx, schedule_config, kind, checkpointing, windowed=True, data=data)

    _assert_close(cut, whole)
