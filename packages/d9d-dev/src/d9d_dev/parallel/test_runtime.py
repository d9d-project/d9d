import zlib

import pytest
import torch
import torch.distributed as dist
from d9d.dist_context import REGULAR_DOMAIN, DeviceMeshParameters
from d9d.models import DecoderForCausalLM, parallelize_causal_lm_head
from d9d.models.io import SequenceHeadShared, SequenceInput, SequenceShared
from d9d.models.qwen3_dense import (
    Qwen3DenseLayerParameters,
    Qwen3DenseModel,
    Qwen3DenseParameters,
    parallelize_qwen3_dense_model,
)
from d9d.models.qwen3_moe import (
    Qwen3MoELayerParameters,
    Qwen3MoEModel,
    Qwen3MoEParameters,
    parallelize_qwen3_moe_model,
)
from d9d.nn.head import SequenceCausalLMHeadShared
from d9d.nn.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.pipelining.api import PipelineStageInfo
from d9d.pipelining.factory import (
    PipelineSchedule1F1BConfig,
    PipelineScheduleDualPipeVConfig,
    PipelineScheduleGPipeConfig,
    PipelineScheduleLoopedBFSConfig,
    PipelineScheduleZeroBubbleVConfig,
    build_program_builder,
    build_schedule,
)
from torch.distributed.tensor import Shard

from d9d_dev.parallel.runtime_step import make_runtime

_LAYER = Qwen3DenseLayerParameters(
    hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-5, head_dim=16
)
_PARAMS = Qwen3DenseParameters(
    layer=_LAYER,
    num_hidden_layers=8,
    rope_base=10000,
    max_position_ids=128,
    split_vocab_size={"a": 50},
    split_vocab_order=["a"],
)
_MOE_PARAMS = Qwen3MoEParameters(
    layer=Qwen3MoELayerParameters(
        hidden_size=64,
        intermediate_size=32,
        num_experts=8,
        experts_top_k=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        head_dim=16,
    ),
    num_hidden_layers=8,
    rope_base=10000,
    max_position_ids=128,
    split_vocab_size={"a": 50},
    split_vocab_order=["a"],
)
_BATCH, _SEQ = 2, 8


def _global_weight(name, param):
    torch.manual_seed(zlib.crc32(name.encode()))
    return (0.02 * torch.randn(param.shape, device="cuda") + 0.1).to(param.dtype)


def _batches(replica):
    inputs, shared = [], []
    for mb in range(4):
        torch.manual_seed(100 * replica + mb)
        ids = torch.randint(0, 50, (_BATCH, _SEQ), device="cuda")
        positions = torch.arange(_SEQ, device="cuda").expand(_BATCH, _SEQ)
        inputs.append(SequenceInput(input_ids=ids))
        shared.append(
            SequenceHeadShared(
                sequence=SequenceShared(position_ids=positions), head=SequenceCausalLMHeadShared(labels=ids)
            )
        )
    return inputs, shared


def _one_gpu_gradients(num_replicas, kind, checkpointing, bf16=True):
    stage = PipelineStageInfo(current_stage=0, num_stages=1)
    if kind == "moe":
        backbone = Qwen3MoEModel(_MOE_PARAMS, stage, HiddenStatesAggregationMode.no, checkpointing)
    else:
        backbone = Qwen3DenseModel(_PARAMS, stage, HiddenStatesAggregationMode.no, checkpointing)
    model = DecoderForCausalLM(backbone, stage)
    model.to_empty(device="cuda")
    model.reset_parameters()
    if bf16:
        model = model.bfloat16()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(_global_weight(name, param))
    for param in model.parameters():
        param.grad_dtype = torch.float32
    for replica in range(num_replicas):
        for one_input, one_shared in zip(*_batches(replica), strict=True):
            model(one_input, one_shared).cross_entropy.sum().backward()
    return {name: param.grad.clone() for name, param in model.named_parameters()}


def _local(param):
    return param.data.to_local()


def _shard_of(full, param):
    for dim, placement in enumerate(param.placements):
        if isinstance(placement, Shard):
            mesh = param.device_mesh
            sub_mesh = mesh[mesh.mesh_dim_names[dim]]
            rows = -(-full.shape[placement.dim] // sub_mesh.size())
            return full.narrow(placement.dim, rows * sub_mesh.get_local_rank(), _local(param).shape[placement.dim])
    return full


def _run(dist_ctx, schedule_config, checkpointing, kind="dense"):
    runtime = make_runtime(bucket_size_mb=1)
    models = {}

    def _provider(stage):
        torch.manual_seed(stage.current_stage)
        if kind == "moe":
            backbone = Qwen3MoEModel(_MOE_PARAMS, stage, HiddenStatesAggregationMode.no, checkpointing)
        else:
            backbone = Qwen3DenseModel(_PARAMS, stage, HiddenStatesAggregationMode.no, checkpointing)
        with torch.device("meta"):
            model = DecoderForCausalLM(backbone, stage)
            # DeepEP supports bf16 hidden states only.
            model = model.bfloat16() if kind == "moe" else model
        parallelize = parallelize_qwen3_moe_model if kind == "moe" else parallelize_qwen3_dense_model
        parallelize(dist_ctx, model.model, stage)
        if stage.is_current_stage_last:
            parallelize_causal_lm_head(model.head, dist_ctx)
        model.to_empty(device="cuda")
        with torch.no_grad():
            model.reset_parameters()
            # DTensor init draws from its own generator, which moves between builds. Both runs need equal weights.
            for name, param in model.named_parameters():
                _local(param).copy_(_shard_of(_global_weight(name, param), param))
        models[stage.current_stage] = model
        return model

    schedule_info, _ = build_schedule(
        pp_mesh=dist_ctx.mesh_for(REGULAR_DOMAIN)["pp"],
        program_builder=build_program_builder(schedule_config),
        model_provider=_provider,
        runtime=runtime,
    )
    runtime.bind()

    # Each data-parallel rank needs its own data, otherwise a wrong reduction can still look right. The ranks
    # of one pipeline need the same data. The pipeline axis is the first mesh dimension.
    replica = dist.get_rank() % (dist.get_world_size() // dist_ctx.mesh_for(REGULAR_DOMAIN)["pp"].size())
    inputs, shared = _batches(replica)

    schedule_info.schedule.step(
        inputs_microbatches=tuple(inputs),
        shared_microbatches=tuple(shared),
        callback=lambda result, _: result.cross_entropy.sum(),
    )
    runtime.wait()
    runtime.scale_grads(torch.tensor(0.5, device="cuda"))
    return models


def _assert_matches_one_gpu(dist_ctx, models, kind, checkpointing, bf16, rtol, atol):
    pipeline_ranks = dist_ctx.mesh_for(REGULAR_DOMAIN)["pp"].size()
    expected = _one_gpu_gradients(dist.get_world_size() // pipeline_ranks, kind, checkpointing, bf16=bf16)
    for model in models.values():
        for name, param in model.named_parameters():
            # The factor is the scale that _run() applies.
            local = param.grad.to_local().float()
            reference = 0.5 * _shard_of(expected[name], param).float()
            assert torch.allclose(local, reference, rtol=rtol, atol=atol + 0.02 * reference.abs().max()), name


@pytest.mark.distributed
@pytest.mark.parametrize(
    "mesh",
    [
        DeviceMeshParameters(pipeline_parallel=2, data_parallel_replicate=4),
        DeviceMeshParameters(pipeline_parallel=2, data_parallel_shard=4),
        DeviceMeshParameters(pipeline_parallel=2, data_parallel_replicate=2, data_parallel_shard=2),
        DeviceMeshParameters(data_parallel_shard=8),
        DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4),
    ],
)
@pytest.mark.parametrize(
    "schedule_config",
    [
        PipelineScheduleGPipeConfig(),
        PipelineSchedule1F1BConfig(num_stages_per_rank=1, zero_bubble=False),
        PipelineSchedule1F1BConfig(num_stages_per_rank=2, zero_bubble=True),
        PipelineScheduleLoopedBFSConfig(num_stages_per_rank=2),
        PipelineScheduleZeroBubbleVConfig(),
        PipelineScheduleDualPipeVConfig(),
    ],
)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_runtime_matches_one_gpu(dist_ctx_factory, mesh, schedule_config, checkpointing):
    dist_ctx = dist_ctx_factory(mesh)
    if mesh.pipeline_parallel == 1 and not isinstance(schedule_config, PipelineScheduleGPipeConfig):
        pytest.skip("One pipeline rank has one schedule.")

    models = _run(dist_ctx, schedule_config, checkpointing)
    _assert_matches_one_gpu(dist_ctx, models, "dense", checkpointing, bf16=False, rtol=1e-3, atol=1e-4)


@pytest.mark.distributed
@pytest.mark.parametrize(
    "mesh",
    [
        DeviceMeshParameters(data_parallel_replicate=8, expert_parallel=8),
        DeviceMeshParameters(data_parallel_shard=8, expert_parallel=2),
        DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4, expert_parallel=8),
        DeviceMeshParameters(pipeline_parallel=2, data_parallel_shard=4, expert_parallel=4),
    ],
)
@pytest.mark.parametrize("checkpointing", [False, True])
def test_moe_matches_one_gpu(dist_ctx_factory, mesh, checkpointing):
    dist_ctx = dist_ctx_factory(mesh)
    models = _run(dist_ctx, PipelineScheduleGPipeConfig(), checkpointing, kind="moe")
    # bf16 rounds differently on the two paths.
    _assert_matches_one_gpu(dist_ctx, models, "moe", checkpointing, bf16=True, rtol=5e-2, atol=2e-1)


def _two_steps(dist_ctx, runtime):
    """Runs a step with wrong weights, restores them, and runs the step to check."""
    models = {}

    def _provider(stage):
        backbone = Qwen3DenseModel(_PARAMS, stage, HiddenStatesAggregationMode.no, False)
        with torch.device("meta"):
            model = DecoderForCausalLM(backbone, stage)
        parallelize_qwen3_dense_model(dist_ctx, model.model, stage)
        if stage.is_current_stage_last:
            parallelize_causal_lm_head(model.head, dist_ctx)
        model.to_empty(device="cuda")
        with torch.no_grad():
            model.reset_parameters()
        models[stage.current_stage] = model
        return model

    def _set_weights(scale):
        with torch.no_grad():
            for model in models.values():
                for name, param in model.named_parameters():
                    _local(param).copy_(scale * _shard_of(_global_weight(name, param), param))

    schedule_info, _ = build_schedule(
        pp_mesh=dist_ctx.mesh_for(REGULAR_DOMAIN)["pp"],
        program_builder=build_program_builder(PipelineScheduleGPipeConfig()),
        model_provider=_provider,
        runtime=runtime,
    )
    pipeline_ranks = dist_ctx.mesh_for(REGULAR_DOMAIN)["pp"].size()
    inputs, shared = _batches(dist.get_rank() % (dist.get_world_size() // pipeline_ranks))
    _set_weights(3.0)
    runtime.bind()
    runtime.zero_grad()
    for scale in (3.0, 1.0):
        _set_weights(scale)
        schedule_info.schedule.step(
            inputs_microbatches=tuple(inputs),
            shared_microbatches=tuple(shared),
            callback=lambda result, _: result.cross_entropy.sum(),
        )
        runtime.wait()
        if scale != 1.0:
            runtime.zero_grad()
    return models


@pytest.mark.distributed
@pytest.mark.parametrize(
    "mesh",
    [
        DeviceMeshParameters(data_parallel_replicate=8),
        DeviceMeshParameters(data_parallel_shard=8),
        DeviceMeshParameters(pipeline_parallel=2, data_parallel_replicate=2, data_parallel_shard=2),
    ],
)
@pytest.mark.parametrize("budget_mib", [None, 8])
def test_float_storage_with_bf16_compute(dist_ctx_factory, mesh, budget_mib):
    dist_ctx = dist_ctx_factory(mesh)
    pipeline_ranks = dist_ctx.mesh_for(REGULAR_DOMAIN)["pp"].size()
    budget = None if budget_mib is None else budget_mib * 1024 * 1024
    runtime = make_runtime(compute_dtype=torch.bfloat16, memory_budget_bytes=budget)

    models = _two_steps(dist_ctx, runtime)
    expected = _one_gpu_gradients(dist.get_world_size() // pipeline_ranks, "dense", False)

    for model in models.values():
        for name, param in model.named_parameters():
            assert param.dtype == torch.float32, name
            local = param.grad.to_local()
            assert local.dtype == torch.float32, name
            reference = _shard_of(expected[name], param)
            assert torch.allclose(local, reference, rtol=5e-2, atol=2e-1 + 0.02 * reference.abs().max()), name
