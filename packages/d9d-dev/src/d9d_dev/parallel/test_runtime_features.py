import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import DENSE_DOMAIN, REGULAR_DOMAIN, DeviceMeshParameters
from d9d.models import DecoderForCausalLM, parallelize_causal_lm_head
from d9d.models.qwen3_dense import Qwen3DenseModel, parallelize_qwen3_dense_model
from d9d.nn.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.parallel import parallelize_hsdp
from d9d.pipelining.api import PipelineStageInfo, StageSegment, run_segments
from d9d.pipelining.factory import (
    PipelineScheduleGPipeConfig,
    PipelineScheduleInferenceConfig,
    build_program_builder,
    build_schedule,
)
from torch import nn

from d9d_dev.parallel.runtime_step import make_runtime
from d9d_dev.parallel.test_runtime import _PARAMS, _batches, _global_weight, _local, _shard_of

_MESHES = [
    DeviceMeshParameters(data_parallel_shard=8),
    DeviceMeshParameters(data_parallel_replicate=8),
    DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4),
]


class _Plain(nn.Module):
    """A model that describes no segments, as a user can write it."""

    def __init__(self):
        super().__init__()
        self.first = nn.Linear(16, 32)
        self.second = nn.Linear(32, 8)

    def reset_parameters(self):
        self.first.reset_parameters()
        self.second.reset_parameters()

    def forward(self, inputs, shared):
        return self.second(torch.relu(self.first(inputs))) * shared


class _Tied(nn.Module):
    """Three segments, where the head of the last one reuses the embedding of the first one."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(48, 16)
        self.middle = nn.Linear(16, 16)
        self.head = nn.Linear(16, 48, bias=False)
        self.head.weight = self.embed.weight

    def reset_parameters(self):
        self.embed.reset_parameters()
        self.middle.reset_parameters()

    def prepare_segment_constants(self, shared):
        return shared

    def stage_segments(self):
        return (
            StageSegment(name="embed", modules=(self.embed,), call=lambda ids, _: self.embed(ids)),
            StageSegment(name="middle", modules=(self.middle,), call=lambda x, _: torch.relu(self.middle(x))),
            StageSegment(name="head", modules=(self.head,), call=lambda x, scale: self.head(x) * scale),
        )

    def forward(self, inputs, shared):
        return run_segments(self, inputs, shared)


def _tied_data(rank):
    data = []
    for microbatch in range(4):
        torch.manual_seed(1000 * rank + microbatch)
        data.append(torch.randint(0, 48, (4, 6), device="cuda"))
    return data


def _plain_data(rank):
    data = []
    for microbatch in range(4):
        torch.manual_seed(1000 * rank + microbatch)
        data.append(torch.randn(4, 16, device="cuda"))
    return data


def _reference(model_type, data, world):
    model = model_type().cuda()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(_global_weight(name, param))
    for rank in range(world):
        for inputs in data(rank):
            model(inputs, 2.0).square().sum().backward()
    return {name: param.grad for name, param in model.named_parameters()}


def _build(dist_ctx, provider, config, runtime):
    return build_schedule(
        pp_mesh=dist_ctx.mesh_for(REGULAR_DOMAIN)["pp"],
        program_builder=build_program_builder(config),
        model_provider=provider,
        runtime=runtime,
    )


def _set_weights(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            _local(param).copy_(_shard_of(_global_weight(name, param), param))


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", _MESHES)
@pytest.mark.parametrize(("model_type", "data"), [(_Plain, _plain_data), (_Tied, _tied_data)], ids=["plain", "tied"])
# The smallest budget keeps no weights, so each segment gathers again.
@pytest.mark.parametrize("budget", [None, 1], ids=["kept", "regathered"])
def test_own_model(dist_ctx_factory, mesh, model_type, data, budget):
    dist_ctx = dist_ctx_factory(mesh)
    runtime = make_runtime(memory_budget_bytes=budget)
    built = {}

    def _provider(stage):
        with torch.device("meta"):
            model = model_type()
        layout = dist_ctx.mesh_for(DENSE_DOMAIN)["dp_replicate", "dp_cp_shard", "cp_replicate"]
        parallelize_hsdp(model, mesh=layout)
        model.to_empty(device="cuda")
        _set_weights(model)
        built["model"] = model
        return model

    info, _ = _build(dist_ctx, _provider, PipelineScheduleGPipeConfig(), runtime)
    runtime.bind()
    for _ in range(2):
        runtime.zero_grad()
        info.schedule.step(
            inputs_microbatches=tuple(data(dist.get_rank())),
            shared_microbatches=(2.0,) * 4,
            callback=lambda result, _: result.square().sum(),
        )
        runtime.wait()

    reference = _reference(model_type, data, dist.get_world_size())
    for name, param in built["model"].named_parameters():
        expected = _shard_of(reference[name], param)
        assert torch.allclose(param.grad.to_local(), expected, rtol=1e-3, atol=1e-3), name


def _one_gpu_model():
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    backbone = Qwen3DenseModel(_PARAMS, stage, HiddenStatesAggregationMode.no, False)
    model = DecoderForCausalLM(backbone, stage)
    model.to_empty(device="cuda")
    model.reset_parameters()
    model = model.bfloat16()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(_global_weight(name, param))
    return model


def _is_frozen(name):
    # Odd layers stay frozen, so most segments have one kind of parameters and the embedding is frozen too.
    return "embed" in name or any(f"layers.{index}." in name for index in (1, 3, 5, 7))


def _distributed_model(dist_ctx, stage, freeze):
    backbone = Qwen3DenseModel(_PARAMS, stage, HiddenStatesAggregationMode.no, False)
    with torch.device("meta"):
        model = DecoderForCausalLM(backbone, stage).bfloat16()
    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad_(not _is_frozen(name))
    parallelize_qwen3_dense_model(dist_ctx, model.model, stage)
    parallelize_causal_lm_head(model.head, dist_ctx)
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.reset_parameters()
    _set_weights(model)
    return model


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", _MESHES)
def test_frozen_parameters(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    runtime = make_runtime(bucket_size_mb=1, memory_budget_bytes=2**31)
    built = {}

    def _provider(stage):
        built["model"] = _distributed_model(dist_ctx, stage, freeze=True)
        return built["model"]

    info, _ = _build(dist_ctx, _provider, PipelineScheduleGPipeConfig(), runtime)
    runtime.bind()
    inputs, shared = _batches(dist.get_rank())
    for step in range(2):
        runtime.zero_grad()
        info.schedule.step(
            inputs_microbatches=tuple(inputs),
            shared_microbatches=tuple(shared),
            callback=lambda result, _: result.cross_entropy.sum(),
        )
        runtime.wait()
        frozen = [bucket for stage in runtime._stages.values() for bucket in stage.buckets if not bucket.plan.trainable]
        assert all(bucket.weights_ready for bucket in frozen), step

    reference = _one_gpu_model()
    for parameter in reference.parameters():
        parameter.grad_dtype = torch.float32
    for rank in range(dist.get_world_size()):
        for one_input, one_shared in zip(*_batches(rank), strict=True):
            reference(one_input, one_shared).cross_entropy.sum().backward()
    expected = dict(reference.named_parameters())

    for name, param in built["model"].named_parameters():
        if _is_frozen(name):
            assert param.grad is None, name
            continue
        grad = _shard_of(expected[name].grad, param)
        assert torch.allclose(param.grad.to_local().float(), grad.float(), rtol=2e-2, atol=2e-2), name


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", _MESHES)
def test_inference(dist_ctx_factory, mesh, monkeypatch):
    dist_ctx = dist_ctx_factory(mesh)
    runtime = make_runtime(memory_budget_bytes=2**31)
    info, _ = _build(
        dist_ctx,
        lambda stage: _distributed_model(dist_ctx, stage, freeze=False),
        PipelineScheduleInferenceConfig(),
        runtime,
    )
    runtime.bind(training=False)
    inputs, shared = _batches(dist.get_rank())

    calls = []
    original = dist.all_gather_single
    monkeypatch.setattr(dist, "all_gather_single", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    outputs = []
    gathers = []
    with torch.inference_mode():
        for _ in range(2):
            info.schedule.step(
                inputs_microbatches=tuple(inputs),
                shared_microbatches=tuple(shared),
                callback=lambda result, _: outputs.append(result.cross_entropy.detach().clone()),
            )
            gathers.append(len(calls))
    results = [torch.stack(outputs[:4]), torch.stack(outputs[4:])]
    assert gathers[0] > 0 or mesh.data_parallel_shard == 1
    # Without gradients the weights stay valid, so the second pass needs no gather.
    assert gathers[1] == gathers[0]

    reference = _one_gpu_model()
    with torch.no_grad():
        expected = torch.stack([reference(x, y).cross_entropy for x, y in zip(inputs, shared, strict=True)])
    assert torch.allclose(results[0].float(), expected.float(), rtol=1e-2, atol=1e-2)
    assert torch.equal(results[0], results[1])
