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
from d9d.pipelining.stage.segments import validate_segments
from torch import nn

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


def _build_backbone(kind, stage, snapshot, checkpointing, num_layers=6):
    torch.manual_seed(stage.current_stage)
    top = {**_TOP, "num_hidden_layers": num_layers}
    mode = HiddenStatesAggregationMode.mean if snapshot else HiddenStatesAggregationMode.no
    if kind == "moe":
        layer = Qwen3MoELayerParameters(intermediate_size=32, num_experts=4, experts_top_k=2, **_COMMON)
        return Qwen3MoEModel(Qwen3MoEParameters(layer=layer, **top), stage, mode, checkpointing)
    layer = Qwen3DenseLayerParameters(intermediate_size=96, **_COMMON)
    return Qwen3DenseModel(Qwen3DenseParameters(layer=layer, **top), stage, mode, checkpointing)


def _materialize(model):
    model.to_empty(device=_DEVICE)
    with torch.no_grad():
        model.reset_parameters()
    return model


def _middle_inputs(seed, snapshot):
    torch.manual_seed(seed)
    hidden = torch.randn(_BATCH, _SEQ, _HIDDEN, device=_DEVICE, requires_grad=True)
    snap = torch.randn(3, _BATCH, _HIDDEN, device=_DEVICE, requires_grad=True) if snapshot else None
    return SequenceTransfer(hidden_states=hidden, hidden_states_snapshot=snap)


def _shared(snapshot):
    positions = torch.arange(_SEQ, device=_DEVICE).expand(_BATCH, _SEQ)
    mask = torch.ones(_BATCH, _SEQ, device=_DEVICE) if snapshot else None
    return SequenceShared(position_ids=positions, hidden_states_agg_mask=mask)


def _cotangent(outputs, seed):
    torch.manual_seed(seed)
    return SequenceTransfer(
        hidden_states=torch.randn_like(outputs.hidden_states),
        hidden_states_snapshot=None
        if outputs.hidden_states_snapshot is None
        else torch.randn_like(outputs.hidden_states_snapshot),
    )


def _runtime(stage_index, model):
    runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mb=32)
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
        inputs, outputs, chain = forward.pop_inputs_outputs(mb)
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
        torch.manual_seed(30 + mb)
        ids = torch.randint(0, 50, (_BATCH, _SEQ), device=_DEVICE)
        shared = SequenceHeadShared(sequence=_shared(snapshot), head=SequenceCausalLMHeadShared(labels=ids))
        forward.run(mb, SequenceInput(input_ids=ids), shared)

    for mb in reversed(range(2)):
        inputs, outputs, chain = forward.pop_inputs_outputs(mb)
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
        assert torch.allclose(value, right[name], rtol=1e-3, atol=1e-4), name


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
        assert torch.allclose(cut_inputs[mb].hidden_states, whole_inputs[mb].hidden_states, rtol=1e-3, atol=1e-5)
        if snapshot:
            assert torch.allclose(
                cut_inputs[mb].hidden_states_snapshot, whole_inputs[mb].hidden_states_snapshot, rtol=1e-3, atol=1e-5
            )


@pytest.mark.local
@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_single_stage_with_head_matches_one_graph(kind, checkpointing, split):
    whole = _run_single_stage(kind, False, checkpointing, split, segmented=False)
    cut = _run_single_stage(kind, False, checkpointing, split, segmented=True)

    _assert_close(cut, whole)


class _FanOut(nn.Module):
    """One tensor has two consumers, one tensor passes through a segment, one output stays unused."""

    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)
        self.third = nn.Linear(4, 4)

    def prepare_segment_constants(self, shared):
        return shared

    def stage_segments(self):
        return [
            StageSegment("first", (self.first,), lambda x, _: (self.first(x), self.first(x) * 2)),
            StageSegment("second", (self.second,), lambda c, _: (c[0], c[1], self.second(c[0]))),
            StageSegment("third", (self.third,), lambda c, _: self.third(c[0]) + c[2]),
        ]

    def forward(self, inputs, shared):
        return run_segments(self, inputs, shared)


@pytest.mark.local
def test_fan_out_pass_through_and_unused_output():
    def run(segmented):
        torch.manual_seed(0)
        model = _FanOut()
        runtime = _runtime(1, model) if segmented else None
        forward = ForwardComputeHandler(1, model, runtime=runtime)
        backward = BackwardComputeHandler(1, model, has_input_peer=True, runtime=runtime)
        inputs = torch.randn(2, 4, requires_grad=True)
        forward.run(0, inputs, None)
        stage_inputs, outputs, chain = forward.pop_inputs_outputs(0)
        backward.backward_full(0, stage_inputs, BackwardSeedLoss(loss=outputs.sum()), chain=chain)
        return {n: p.grad.clone() for n, p in model.named_parameters()}, backward.pop_for_sending(0)

    whole, whole_input = run(False)
    cut, cut_input = run(True)

    _assert_close(cut, whole)
    assert torch.allclose(cut_input, whole_input)


@pytest.mark.local
def test_segments_must_own_all_trainable_parameters():
    model = _FanOut()
    segments = model.stage_segments()[:2]

    with pytest.raises(ValueError, match="third"):
        validate_segments(model, segments)


def _run_pipeline(dist_ctx, schedule_config, kind, checkpointing, n_microbatches, segmented):
    models = {}

    def _provider(stage):
        models[stage.current_stage] = _materialize(
            DecoderForCausalLM(_build_backbone(kind, stage, False, checkpointing, num_layers=16), stage)
        )
        return models[stage.current_stage]

    runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mb=32) if segmented else None
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
        torch.manual_seed(40 + mb)
        ids = torch.randint(0, 50, (_BATCH, _SEQ), device=_DEVICE)
        inputs.append(SequenceInput(input_ids=ids))
        shared.append(SequenceHeadShared(sequence=_shared(False), head=SequenceCausalLMHeadShared(labels=ids)))

    schedule_info.schedule.step(
        inputs_microbatches=tuple(inputs),
        shared_microbatches=tuple(shared),
        callback=lambda result, _: result.cross_entropy.sum(),
    )

    for stage_object in schedule_info.schedule._stages.values():
        assert len(stage_object._forward_comp._cache) == 0
        assert len(stage_object._backward_comp._cache) == 0

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
