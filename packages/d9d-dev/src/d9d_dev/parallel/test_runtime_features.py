import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import REGULAR_DOMAIN
from d9d.models import DecoderForCausalLM
from d9d.models.qwen3_dense import Qwen3DenseModel
from d9d.nn.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.pipelining.api import PipelineStageInfo, StageSegment, run_segments
from d9d.pipelining.factory import (
    PipelineScheduleGPipeConfig,
    PipelineScheduleInferenceConfig,
    build_program_builder,
    build_schedule,
)
from torch import nn

from d9d_dev.nn.helper.tolerances import forward_tolerance_for
from d9d_dev.parallel.runtime_step import (
    MESHES,
    PARAMS,
    batches,
    global_weight,
    local,
    make_runtime,
    place_hsdp,
    shard_of,
)
from d9d_dev.pipelining.step_handler import FunctionStepHandler

_MESHES = MESHES


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


class _SegmentModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs, constants):
        return self.module(inputs)


class _ScaledSegmentModule(_SegmentModule):
    def forward(self, inputs, constants):
        return super().forward(inputs, constants) * constants


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
            StageSegment(name="embed", module=_SegmentModule(self.embed)),
            StageSegment(name="middle", module=_SegmentModule(nn.Sequential(self.middle, nn.ReLU()))),
            StageSegment(name="head", module=_ScaledSegmentModule(self.head)),
        )

    def forward(self, inputs, shared):
        return run_segments(self, inputs, shared)


def _generator(rank, microbatch):
    # The reference rebuilds the data of every rank.
    return torch.Generator(device="cuda").manual_seed(1000 * rank + microbatch)


def _tied_data(rank):
    return [torch.randint(0, 48, (4, 6), device="cuda", generator=_generator(rank, mb)) for mb in range(4)]


def _plain_data(rank):
    return [torch.randn(4, 16, device="cuda", generator=_generator(rank, mb)) for mb in range(4)]


def _reference(model_type, data, world):
    model = model_type().cuda()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(global_weight(name, param))
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
            local(param).copy_(shard_of(global_weight(name, param), param))


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", _MESHES)
@pytest.mark.parametrize(("model_type", "data"), [(_Plain, _plain_data), (_Tied, _tied_data)], ids=["plain", "tied"])
# The smallest budget keeps no weights, so each segment gathers again.
@pytest.mark.parametrize("budget", [2**31, 0], ids=["kept", "regathered"])
def test_own_model(dist_ctx_factory, mesh, model_type, data, budget):
    dist_ctx = dist_ctx_factory(mesh)
    runtime = make_runtime(memory_budget_bytes=budget)
    built = {}

    def _provider(stage):
        with torch.device("meta"):
            model = model_type()
        place_hsdp(model, dist_ctx)
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
            handler=FunctionStepHandler(lambda result, _: result.square().sum()),
        )
        runtime.wait()

    reference = _reference(model_type, data, dist.get_world_size())
    tolerance = forward_tolerance_for(torch.float32)
    for name, param in built["model"].named_parameters():
        expected = shard_of(reference[name], param)
        torch.testing.assert_close(
            param.grad.to_local(),
            expected,
            rtol=tolerance.rtol,
            atol=tolerance.atol,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def _one_gpu_model():
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    backbone = Qwen3DenseModel(PARAMS, stage, HiddenStatesAggregationMode.no, False)
    model = DecoderForCausalLM(backbone, stage)
    model.to_empty(device="cuda")
    model.reset_parameters()
    model = model.bfloat16()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(global_weight(name, param))
    return model


def _is_frozen(name):
    # Odd layers stay frozen, so most segments have one kind of parameters and the embedding is frozen too.
    return "embed" in name or any(f"layers.{index}." in name for index in (1, 3, 5, 7))


def _distributed_model(dist_ctx, stage, freeze):
    backbone = Qwen3DenseModel(PARAMS, stage, HiddenStatesAggregationMode.no, False)
    with torch.device("meta"):
        model = DecoderForCausalLM(backbone, stage).bfloat16()
    if freeze:
        for name, param in model.named_parameters():
            param.requires_grad_(not _is_frozen(name))
    place_hsdp(model, dist_ctx)
    model.to_empty(device="cuda")
    with torch.no_grad():
        model.reset_parameters()
    _set_weights(model)
    return model


@pytest.mark.distributed
@pytest.mark.parametrize("mesh", _MESHES)
def test_frozen_parameters(dist_ctx_factory, mesh):
    dist_ctx = dist_ctx_factory(mesh)
    runtime = make_runtime(bucket_size_mib=1, memory_budget_bytes=2**31)
    built = {}

    def _provider(stage):
        built["model"] = _distributed_model(dist_ctx, stage, freeze=True)
        return built["model"]

    info, _ = _build(dist_ctx, _provider, PipelineScheduleGPipeConfig(), runtime)
    runtime.bind()
    inputs, shared = batches(dist.get_rank())
    for _ in range(2):
        runtime.zero_grad()
        info.schedule.step(
            inputs_microbatches=tuple(inputs),
            shared_microbatches=tuple(shared),
            handler=FunctionStepHandler(lambda result, _: result.cross_entropy.sum()),
        )
        runtime.wait()

    reference = _one_gpu_model()
    for parameter in reference.parameters():
        parameter.grad_dtype = torch.float32
    for rank in range(dist.get_world_size()):
        for one_input, one_shared in zip(*batches(rank), strict=True):
            reference(one_input, one_shared).cross_entropy.sum().backward()
    expected = dict(reference.named_parameters())

    tolerance = forward_tolerance_for(torch.bfloat16)
    for name, param in built["model"].named_parameters():
        if _is_frozen(name):
            assert param.grad is None, name
            continue
        grad = shard_of(expected[name].grad, param)
        # The gradients sum in bf16 here and in fp32 in the reference. The measured gap is 4.4e-3 above rtol.
        torch.testing.assert_close(
            param.grad.to_local().float(),
            grad.float(),
            rtol=tolerance.rtol,
            atol=5e-3,
            msg=lambda message, name=name: f"{name}: {message}",
        )


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
    inputs, shared = batches(dist.get_rank())

    # The runtime exposes no gather count, and the profiler names of collectives change between PyTorch versions.
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
                handler=FunctionStepHandler(lambda result, _: outputs.append(result.cross_entropy.detach().clone())),
            )
            gathers.append(len(calls))
    results = [torch.stack(outputs[:4]), torch.stack(outputs[4:])]
    assert gathers[0] > 0 or mesh.data_parallel_shard == 1
    # Without gradients the weights stay valid, so the second pass needs no gather.
    assert gathers[1] == gathers[0]

    reference = _one_gpu_model()
    with torch.no_grad():
        expected = torch.stack([reference(x, y).cross_entropy for x, y in zip(inputs, shared, strict=True)])
    tolerance = forward_tolerance_for(torch.bfloat16)
    torch.testing.assert_close(results[0].float(), expected.float(), rtol=tolerance.rtol, atol=tolerance.atol)
    assert torch.equal(results[0], results[1])
