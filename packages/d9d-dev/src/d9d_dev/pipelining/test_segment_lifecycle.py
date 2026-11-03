from contextlib import contextmanager
from unittest.mock import Mock
from weakref import ref

import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import DATA_SHARD_DIMS, DENSE_DOMAIN, DeviceMeshParameters
from d9d.parallel import parallelize_hsdp
from d9d.parallel.runtime import BucketId, ParameterRuntime, ReductionPlan
from d9d.pipelining.api import PipelineStageInfo, StageSegment, run_segments
from d9d.pipelining.factory import PipelineScheduleInferenceConfig, build_program_builder
from d9d.pipelining.runtime.executor import PipelineScheduleExecutor
from d9d.pipelining.runtime.offline import OfflinePipelineExecutor
from d9d.pipelining.stage import PipelineStage
from d9d.pipelining.stage.segments import SegmentChain, backward_chains, forward_chains, validate_segments
from torch import nn
from torch.distributed.tensor import distribute_tensor


class _RecordingRuntime:
    def __init__(self):
        self.bindings = []
        self.plans = []
        self.reductions = []

    @contextmanager
    def segment_weights(self, stage_index, segment_index, writes_grads, next_segment=None):
        self.bindings.append((stage_index, segment_index, writes_grads))
        yield

    def reduction_plan(self, stage_index, final):
        self.plans.append((stage_index, final))
        return ReductionPlan(final, {0: (BucketId(stage_index, 0),)})

    def reduce(self, bucket_id, final):
        self.reductions.append((bucket_id, final))

    def register_stage(self, stage_index, module):
        pass


def _forward(chain, runtime, inputs, shared):
    chain.start_forward(inputs, shared)
    return forward_chains([chain], runtime, 0)[0]


def _backward_full(chain, runtime, loss, plan=None):
    chain.start_backward([loss], None)
    backward_chains([chain], runtime, 0, plan)


def _backward_input(chain, loss):
    chain.start_backward([loss], None)
    chain.backward_input()


class _OptionalConsumer(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.optional = nn.Linear(4, 4)

    def prepare_segment_constants(self, shared):
        return shared

    def stage_segments(self):
        return [
            StageSegment("first", (self.first,), lambda x, _: self.first(x)),
            StageSegment("optional", (self.optional,), lambda x, _: (x, self.optional(x))),
            StageSegment("select", (), lambda x, use_optional: x[1] if use_optional else x[0]),
        ]

    def forward(self, inputs, shared):
        return run_segments(self, inputs, shared)


@pytest.mark.local
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("use_optional", [False, True])
def test_unused_segments_keep_parameter_bindings_in_order(split, use_optional):
    torch.manual_seed(0)
    model = _OptionalConsumer()
    inputs = torch.randn(2, 4, requires_grad=True)
    model(inputs, use_optional).sum().backward()
    expected = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    expected_input = inputs.grad.clone()
    model.zero_grad(set_to_none=True)
    inputs.grad = None

    runtime = _RecordingRuntime()
    chain = SegmentChain(model, model.stage_segments(), 0, runtime)
    loss = _forward(chain, runtime, inputs, use_optional).sum()
    runtime.bindings.clear()
    if split:
        _backward_input(chain, loss)
        assert all(p.grad is None for p in model.parameters())
        assert runtime.bindings == [(0, index, False) for index in (2, 1, 0)]
        runtime.bindings.clear()
        chain.backward_weight(None)
    else:
        _backward_full(chain, runtime, loss)

    assert runtime.bindings == [(0, index, True) for index in (2, 1, 0)]
    torch.testing.assert_close(chain.input_grads([inputs])[0], expected_input)
    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


class _ParameterOnlySegment(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 4)
        self.projection = nn.Linear(4, 4)

    def prepare_segment_constants(self, shared):
        return shared

    def stage_segments(self):
        return [
            StageSegment("embedding", (self.embedding,), lambda x, _: self.embedding(x)),
            StageSegment("projection", (self.projection,), lambda x, _: self.projection(x)),
        ]


@pytest.mark.local
def test_split_backward_defers_segment_without_differentiable_inputs():
    torch.manual_seed(0)
    model = _ParameterOnlySegment()
    inputs = torch.tensor([0, 2])
    run_segments(model, inputs, None).sum().backward()
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)

    runtime = _RecordingRuntime()
    chain = SegmentChain(model, model.stage_segments(), 0, runtime)
    _backward_input(chain, _forward(chain, runtime, inputs, None).sum())
    assert all(p.grad is None for p in model.parameters())
    chain.backward_weight(None)

    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


@pytest.mark.local
def test_offline_schedule_reduces_each_microbatch_and_marks_only_the_last_final():
    model = _OptionalConsumer()
    runtime = _RecordingRuntime()
    schedule = OfflinePipelineExecutor(model, do_backward=True, runtime=runtime)

    schedule.step((torch.randn(2, 4), torch.randn(2, 4)), (True, False), lambda output, _: output.sum())

    assert runtime.plans == [(0, False), (0, True)]
    assert runtime.reductions == [(BucketId(0, 0), False), (BucketId(0, 0), True)]


@pytest.mark.local
def test_segments_must_own_frozen_parameters():
    model = _OptionalConsumer()
    model.optional.requires_grad_(False)

    with pytest.raises(ValueError, match="optional"):
        validate_segments(model, model.stage_segments()[:1])


@pytest.mark.local
def test_segment_constants_cannot_carry_gradients():
    model = _OptionalConsumer()
    chain = SegmentChain(model, model.stage_segments(), 0, _RecordingRuntime())

    with pytest.raises(ValueError, match="constants contain a tensor with a gradient"):
        chain.start_forward(torch.randn(2, 4), torch.ones((), requires_grad=True))


@pytest.mark.local
@pytest.mark.parametrize("with_runtime", [False, True])
def test_inference_releases_forward_cache_after_each_pack(with_runtime):
    model = _OptionalConsumer().eval()
    runtime = _RecordingRuntime() if with_runtime else None
    group = Mock()
    group.size.return_value = 1
    group.rank.return_value = 0
    stage = PipelineStage(PipelineStageInfo(current_stage=0, num_stages=1), model, group, {0: 0}, runtime=runtime)
    schedule = PipelineScheduleExecutor(
        group, [stage], build_program_builder(PipelineScheduleInferenceConfig()), runtime=runtime
    )
    outputs = []

    with torch.no_grad():
        for count in (3, 1):
            schedule.step(
                tuple(torch.randn(2, 4) for _ in range(count)),
                (True,) * count,
                lambda output, _: outputs.append(ref(output)),
            )
            assert stage._forward_comp._cache == {}
            assert all(reference() is None for reference in outputs)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "mesh",
    [
        DeviceMeshParameters(data_parallel_replicate=8),
        DeviceMeshParameters(data_parallel_shard=8),
        DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4),
    ],
)
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_rank_dependent_unused_segments_reduce_like_one_model(dist_ctx_factory, mesh, split, tied):
    dist_ctx = dist_ctx_factory(mesh)
    model = _OptionalConsumer().cuda()
    reference = _OptionalConsumer().cuda()
    if tied:
        model.optional.weight = model.first.weight
        reference.optional.weight = reference.first.weight
    reference.load_state_dict(model.state_dict())
    parallelize_hsdp(model, dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate", "dp_cp_shard", "cp_replicate"])
    runtime = ParameterRuntime(gather_dims=DATA_SHARD_DIMS, bucket_size_mb=32, memory_budget_bytes=1)
    runtime.register_stage(0, model)
    runtime.bind()

    def inputs(rank, microbatch):
        return torch.full((2, 4), (rank + microbatch + 1) / 8, device="cuda", requires_grad=True)

    try:
        for microbatch in range(2):
            chain = SegmentChain(model, model.stage_segments(), 0, runtime)
            loss = _forward(chain, runtime, inputs(dist.get_rank(), microbatch), dist.get_rank() % 2 == 0).sum()
            plan = runtime.reduction_plan(0, final=microbatch == 1)
            if split:
                _backward_input(chain, loss)
                chain.backward_weight(plan)
            else:
                _backward_full(chain, runtime, loss, plan)
        runtime.wait()

        for rank in range(dist.get_world_size()):
            for microbatch in range(2):
                reference(inputs(rank, microbatch), rank % 2 == 0).sum().backward()
        for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
            gradient = distribute_tensor(expected.grad, parameter.device_mesh, parameter.placements).to_local()
            torch.testing.assert_close(parameter.grad.to_local(), gradient, rtol=1e-4, atol=1e-4)
    finally:
        runtime.unbind()
