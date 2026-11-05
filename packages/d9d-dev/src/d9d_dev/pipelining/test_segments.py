from weakref import ref

import pytest
import torch
from d9d.dist_context import DATA_SHARD_DIMS, REGULAR_DOMAIN, DeviceMeshParameters
from d9d.models import DecoderForCausalLM
from d9d.models.io import SequenceHeadShared, SequenceInput, SequenceShared, SequenceTransfer
from d9d.models.qwen3_dense import Qwen3DenseLayerParameters, Qwen3DenseModel, Qwen3DenseParameters
from d9d.models.qwen3_moe import Qwen3MoELayerParameters, Qwen3MoEModel, Qwen3MoEParameters
from d9d.nn.head import SequenceCausalLMHeadShared
from d9d.nn.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.parallel.runtime import ParameterRuntime
from d9d.parallel.runtime.plan import check_segments
from d9d.pipelining.api import PipelineStageInfo, StageSegment, run_segments
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

from d9d_dev.nn.helper.tolerances import forward_tolerance_for
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
_TOLERANCE = forward_tolerance_for(torch.float32)


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


def _generator(seed):
    return torch.Generator(device=_DEVICE).manual_seed(seed)


def _randn(shape, generator):
    return torch.randn(shape, device=_DEVICE, generator=generator)


def _middle_inputs(seed, snapshot):
    generator = _generator(seed)
    hidden = _randn((_BATCH, _SEQ, _HIDDEN), generator).requires_grad_()
    snap = _randn((3, _BATCH, _HIDDEN), generator).requires_grad_() if snapshot else None
    return SequenceTransfer(hidden_states=hidden, hidden_states_snapshot=snap)


def _shared(snapshot):
    positions = torch.arange(_SEQ, device=_DEVICE).expand(_BATCH, _SEQ)
    mask = torch.ones(_BATCH, _SEQ, device=_DEVICE) if snapshot else None
    return SequenceShared(position_ids=positions, hidden_states_agg_mask=mask)


def _cotangent(outputs, seed):
    generator = _generator(seed)
    snapshot = outputs.hidden_states_snapshot
    return SequenceTransfer(
        hidden_states=_randn(outputs.hidden_states.shape, generator),
        hidden_states_snapshot=None if snapshot is None else _randn(snapshot.shape, generator),
    )


def _ids(seed):
    return torch.randint(0, 50, (_BATCH, _SEQ), device=_DEVICE, generator=_generator(seed))


def _runtime(stage_index, model):
    runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mib=32)
    runtime.register_stage(stage_index, model)
    runtime.bind()
    return runtime


def _run_middle_stage(kind, snapshot, checkpointing, split, segmented):
    """Runs two microbatches on a middle stage: F0, F1, then B1, B0 or I1, I0, W1, W0."""
    stage = PipelineStageInfo(current_stage=1, num_stages=3)
    model = _materialize(_build_backbone(kind, stage, snapshot, checkpointing))
    runtime = _runtime(1, model) if segmented else None
    forward = ForwardComputeHandler(1, model, runtime=runtime)
    backward = BackwardComputeHandler(1, model, has_input_peer=True, runtime=runtime)
    shared = _shared(snapshot)

    for mb in range(2):
        forward.run(mb, _middle_inputs(10 + mb, snapshot), shared)

    input_grads = {}
    for mb in reversed(range(2)):
        cache = forward.pop_cache(mb)
        inputs, outputs, chain = cache.inputs, cache.outputs, cache.chain
        assert (chain is not None) == segmented
        seed = BackwardSeedTransfer(outputs=outputs, output_grads=_cotangent(outputs, 20 + mb))
        if split:
            backward.backward_input(mb, inputs, seed, chain=chain)
        else:
            backward.backward_full(mb, inputs, seed, chain=chain)
        input_grads[mb] = backward.pop_for_sending(mb)
    if split:
        for mb in reversed(range(2)):
            backward.backward_weight(mb)

    return {n: p.grad.clone() for n, p in model.named_parameters()}, input_grads


def _run_single_stage(kind, snapshot, checkpointing, split, segmented):
    """Runs two microbatches of the whole model with a head, from token ids to the loss."""
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    backbone = _build_backbone(kind, stage, snapshot, checkpointing)
    model = _materialize(DecoderForCausalLM(backbone, stage))
    runtime = _runtime(0, model) if segmented else None
    forward = ForwardComputeHandler(0, model, runtime=runtime)
    backward = BackwardComputeHandler(0, model, has_input_peer=False, runtime=runtime)

    for mb in range(2):
        ids = _ids(30 + mb)
        shared = SequenceHeadShared(sequence=_shared(snapshot), head=SequenceCausalLMHeadShared(labels=ids))
        forward.run(mb, SequenceInput(input_ids=ids), shared)

    for mb in reversed(range(2)):
        cache = forward.pop_cache(mb)
        inputs, outputs, chain = cache.inputs, cache.outputs, cache.chain
        assert (chain is not None) == segmented
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
    whole, whole_inputs = _run_middle_stage(kind, snapshot, checkpointing, split, segmented=False)
    cut, cut_inputs = _run_middle_stage(kind, snapshot, checkpointing, split, segmented=True)

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
    whole = _run_single_stage(kind, False, checkpointing, split, segmented=False)
    cut = _run_single_stage(kind, False, checkpointing, split, segmented=True)

    _assert_close(cut, whole)


class _FanOutFirst(nn.Linear):
    def forward(self, activations, constants):
        return super().forward(activations), super().forward(activations) * 2


class _FanOutSecond(nn.Linear):
    def forward(self, activations, constants):
        return activations[0], activations[1], super().forward(activations[0])


class _FanOutThird(nn.Linear):
    def forward(self, activations, constants):
        return super().forward(activations[0]) + activations[2]


class _FanOut(nn.Module):
    """One tensor has two consumers, one tensor passes through a segment, one output stays unused."""

    def __init__(self):
        super().__init__()
        self.first = _FanOutFirst(4, 4)
        self.second = _FanOutSecond(4, 4)
        self.third = _FanOutThird(4, 4)

    def prepare_segment_constants(self, shared):
        return shared

    def stage_segments(self):
        return [
            StageSegment("first", self.first),
            StageSegment("second", self.second),
            StageSegment("third", self.third),
        ]

    def forward(self, inputs, shared):
        return run_segments(self, inputs, shared)


@pytest.mark.local
def test_fan_out_pass_through_and_unused_output():
    def run(segmented):
        model = _set_weights(_FanOut())
        runtime = _runtime(1, model) if segmented else None
        forward = ForwardComputeHandler(1, model, runtime=runtime)
        backward = BackwardComputeHandler(1, model, has_input_peer=True, runtime=runtime)
        inputs = torch.randn(2, 4, generator=torch.Generator().manual_seed(0), requires_grad=True)
        forward.run(0, inputs, None)
        cache = forward.pop_cache(0)
        stage_inputs, outputs, chain = cache.inputs, cache.outputs, cache.chain
        backward.backward_full(0, stage_inputs, BackwardSeedLoss(loss=outputs.sum()), chain=chain)
        return {n: p.grad.clone() for n, p in model.named_parameters()}, backward.pop_for_sending(0)

    whole, whole_input = run(False)
    cut, cut_input = run(True)

    _assert_close(cut, whole)
    torch.testing.assert_close(cut_input, whole_input, rtol=_TOLERANCE.rtol, atol=_TOLERANCE.atol)


@pytest.mark.local
def test_segments_must_own_all_trainable_parameters():
    model = _FanOut()
    segments = model.stage_segments()[:2]

    with pytest.raises(ValueError, match="third"):
        check_segments(model, segments)


def _run_pipeline(dist_ctx, schedule_config, kind, checkpointing, n_microbatches, segmented):
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

    runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mib=32) if segmented else None
    schedule_info, _ = build_schedule(
        pp_mesh=dist_ctx.mesh_for(REGULAR_DOMAIN)["pp"],
        program_builder=build_program_builder(schedule_config),
        model_provider=_provider,
        runtime=runtime,
    )
    if segmented:
        runtime.bind()

    inputs, shared = [], []
    for mb in range(n_microbatches):
        ids = _ids(40 + mb)
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

    whole = _run_pipeline(dist_ctx, schedule_config, kind, checkpointing, 16, segmented=False)
    cut = _run_pipeline(dist_ctx, schedule_config, kind, checkpointing, 16, segmented=True)

    _assert_close(cut, whole)
