import pytest
import torch
import torch.distributed as dist
from d9d.core.dist_context import BATCH_DOMAIN, DENSE_DOMAIN, DeviceMeshParameters, DistributedContext
from d9d.module.parallelism.model.qwen3_moe import parallelize_qwen3_moe_for_causal_lm
from d9d.pipelining.api import PipelineShardingSpec, PipelineStageInfo
from d9d.pipelining.factory import (
    PipelineScheduleGPipeConfig,
    build_schedule,
)
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from d9d_test.modules.checkers import check_grad_distance_all_local_dist, copy_params_local_to_dist
from d9d_test.modules.grad_sync import sync_grads_manually
from d9d_test.modules.model.qwen3_moe.factory import build_decoder, build_decoder_inputs_hf, build_decoder_inputs_my
from d9d_test.modules.model.qwen3_moe.util import check_lm_model_grad, clone_lm_model_weights


def build_hf_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(131231)

    hf = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=100,
            num_hidden_layers=12,
            hidden_size=512,
            moe_intermediate_size=256,
            num_experts=8,
            num_experts_per_tok=7,
            num_attention_heads=16,
            num_key_value_heads=4,
            hidden_act="silu",
            max_position_embeddings=10000,
            rms_norm_eps=1e-5,
            use_cache=False,
            tie_word_embeddings=False,
            rope_theta=10_000,
            attention_bias=False,
            use_sliding_window=False,
            attention_dropout=0.0,
            norm_topk_prob=True,
            router_aux_loss_coef=0.0,
            _attn_implementation="sdpa"
        )
    ).cuda()
    # no rope conversion here
    hf.model.embed_tokens.bfloat16()
    hf.model.layers.bfloat16()
    hf.model.norm.bfloat16()
    hf.lm_head.bfloat16()

    return hf


@pytest.mark.local
@pytest.mark.parametrize("checkpointing", [True, False])
def test_consistent_to_hf(checkpointing):
    stage_info = PipelineStageInfo(current_stage=0, num_stages=1)

    hf = build_hf_model()
    my = build_decoder(stage=stage_info, checkpointing=checkpointing)

    clone_lm_model_weights(my=my, hf=hf, stage=stage_info)

    input_ids_hf, position_ids_hf, labels_hf = build_decoder_inputs_hf()
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        outs_hf = hf(
            input_ids=input_ids_hf,
            position_ids=position_ids_hf,
            labels=labels_hf,
        )

    input_ids_my, position_ids_my, labels_my = build_decoder_inputs_my()
    outs_my = my(input_ids=input_ids_my, position_ids=position_ids_my, labels=labels_my)
    loss_my = outs_my["logps"][labels_my != -100].mean()

    assert torch.allclose(outs_hf.loss, loss_my, atol=1e-4, rtol=0.001)

    outs_hf.loss.backward()
    loss_my.backward()

    check_lm_model_grad(my, hf, stage=stage_info)


def _build_distributed_model(
        stage: PipelineStageInfo,
        dist_ctx: DistributedContext,
        local_model: nn.Module,
        stage_memo: list[tuple[PipelineStageInfo, Qwen3MoeForCausalLM]],
        checkpointing: bool
):
    model = build_decoder(stage=stage, checkpointing=checkpointing)
    parallelize_qwen3_moe_for_causal_lm(dist_ctx, model, stage)
    copy_params_local_to_dist(local=local_model, dist=model)
    stage_memo.append(model)
    return model


def _shard_virtual_dp(tensor: torch.Tensor, dist_ctx: DistributedContext) -> torch.Tensor:
    return DTensor.from_local(
        tensor,
        device_mesh=dist_ctx.mesh_for(BATCH_DOMAIN)["dp"],
        placements=[Replicate()]
    ).redistribute(placements=(Shard(0),)).to_local()


@pytest.mark.distributed
@pytest.mark.parametrize("checkpointing", [True, False])
@pytest.mark.parametrize("mesh", [
    # PP + DPR / EP
    DeviceMeshParameters(
        pipeline_parallel=2,
        expert_parallel=4,
        tensor_parallel=1,
        context_parallel_shard=1,
        data_parallel_shard=1,
        data_parallel_replicate=4,
        context_parallel_replicate=1
    ),
    # PP + DPS / EP
    DeviceMeshParameters(
        pipeline_parallel=2,
        expert_parallel=4,
        tensor_parallel=1,
        context_parallel_shard=1,
        data_parallel_shard=4,
        data_parallel_replicate=1,
        context_parallel_replicate=1
    ),
    # PP + DPR + DPS / EP
    DeviceMeshParameters(
        pipeline_parallel=2,
        expert_parallel=4,
        tensor_parallel=1,
        context_parallel_shard=1,
        data_parallel_shard=2,
        data_parallel_replicate=2,
        context_parallel_replicate=1
    ),
    # PP + DPR + DPS / EP + R
    DeviceMeshParameters(
        pipeline_parallel=2,
        expert_parallel=2,
        tensor_parallel=1,
        context_parallel_shard=1,
        data_parallel_shard=2,
        data_parallel_replicate=2,
        context_parallel_replicate=1
    ),
])
def test_consistent_to_itself_dist(mesh, checkpointing, dist_ctx_factory):
    dist_ctx = dist_ctx_factory(mesh)

    stage_memo = []

    local = build_decoder(stage=PipelineStageInfo(num_stages=1, current_stage=0), checkpointing=checkpointing)
    input_ids, position_ids, labels = build_decoder_inputs_my()
    loss_div = (labels != -100).sum()
    outs_local = local(input_ids=input_ids, position_ids=position_ids, labels=labels)
    loss_local = outs_local["logps"].sum() / loss_div
    loss_local.backward()

    input_ids_dist = _shard_virtual_dp(input_ids, dist_ctx)
    position_ids_dist = _shard_virtual_dp(position_ids, dist_ctx)
    labels_dist = _shard_virtual_dp(labels, dist_ctx)

    loss_dist_accum = []

    def _loss_fn(outputs, _):
        ret = outputs["logps"].sum() / loss_div
        loss_dist_accum.append(ret.detach())
        return ret

    schedule, _ = build_schedule(
        dist_ctx,
        n_microbatches=2,
        schedule_config=PipelineScheduleGPipeConfig(),
        model_provider=lambda stg: _build_distributed_model(
            stage=stg,
            dist_ctx=dist_ctx,
            stage_memo=stage_memo,
            local_model=local,
            checkpointing=checkpointing
        ),
        loss_fn=_loss_fn
    )

    inputs_dist = {
        "input_ids": input_ids_dist
    }
    kwargs_dist = {
        "position_ids": position_ids_dist,
        "labels": labels_dist
    }

    schedule.schedule.configure_buffers(inputs_dist, kwargs_dist, sharding_spec=PipelineShardingSpec())

    schedule.schedule.step(
        inputs_dist,
        kwargs_dist
    )

    pp_mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["pp"]

    if pp_mesh.get_local_rank() == pp_mesh.size() - 1:
        loss_dist = torch.tensor(loss_dist_accum, device="cuda").sum()
        for group in dist_ctx.mesh_for(DENSE_DOMAIN)[
            "dp_replicate", "dp_cp_shard", "cp_replicate", "tp"
        ].get_all_groups():
            dist.all_reduce(loss_dist, op=dist.ReduceOp.SUM, group=group)
        assert torch.allclose(loss_local, loss_dist, atol=1e-4, rtol=0.001)

    for stage_model in stage_memo:
        sync_grads_manually(stage_model)
        check_grad_distance_all_local_dist(
            local=local,
            dist=stage_model
        )
