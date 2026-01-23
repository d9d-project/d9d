import pytest
import torch
from d9d.core.dist_context import BATCH_DOMAIN, DENSE_DOMAIN, EXPERT_DOMAIN, DeviceMeshParameters, DistributedContext
from d9d.module.parallelism.api import parallelize_expert_parallel, parallelize_replicate
from d9d.pipelining.api import PipelineShardingSpec, PipelineStageInfo
from d9d.pipelining.factory import PipelineScheduleLoopedBFSConfig, build_schedule
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

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

    check_lm_model_grad(my, hf, is_dist=False, stage=stage_info)


def _build_distributed_model(
        stage: PipelineStageInfo,
        hf: Qwen3MoeForCausalLM,
        dist_ctx: DistributedContext,
        stage_memo: list[tuple[PipelineStageInfo, Qwen3MoeForCausalLM]],
        checkpointing: bool
):
    dense_mesh = dist_ctx.mesh_for(DENSE_DOMAIN)
    expert_mesh = dist_ctx.mesh_for(EXPERT_DOMAIN)
    model = build_decoder(stage=stage, checkpointing=checkpointing)
    clone_lm_model_weights(my=model, hf=hf, stage=stage)

    if stage.is_current_stage_first:
        parallelize_replicate(
            model.model.embed_tokens,
            mesh=dense_mesh["dp_replicate"],
        )

    if stage.is_current_stage_last:
        parallelize_replicate(
            model.model.norm,
            mesh=dense_mesh["dp_replicate"],
        )
        parallelize_replicate(
            model.lm_head,
            mesh=dense_mesh["dp_replicate"],
        )

    for layer in model.model.layers.values():
        parallelize_expert_parallel(
            layer.mlp,
            mesh_experts=expert_mesh[["ep_replicate", "ep_shard"]]
        )
        parallelize_replicate(
            layer.self_attn,
            mesh=dense_mesh["dp_replicate"],
        )
        parallelize_replicate(
            layer.input_layernorm,
            mesh=dense_mesh["dp_replicate"],
        )
        parallelize_replicate(
            layer.post_attention_layernorm,
            mesh=dense_mesh["dp_replicate"],
        )

    stage_memo.append((stage, model))

    return model


def _shard_virtual_dp(tensor: torch.Tensor, dist_ctx: DistributedContext) -> DTensor:
    return DTensor.from_local(
        tensor,
        device_mesh=dist_ctx.mesh_for(BATCH_DOMAIN)["dp"],
        placements=[Replicate()]
    ).redistribute(placements=(Shard(0),)).to_local()


@pytest.mark.distributed
@pytest.mark.parametrize("checkpointing", [True, False])
def test_consistent_to_hf_dist(checkpointing):
    dist_ctx = DeviceMeshParameters(
        pipeline_parallel=2,
        expert_parallel=4,
        tensor_parallel=1,
        context_parallel_shard=1,
        data_parallel_shard=1,
        data_parallel_replicate=4,
        context_parallel_replicate=1
    ).build()
    dense_mesh = dist_ctx.mesh_for(DENSE_DOMAIN)

    hf = build_hf_model()
    stage_memo = []

    input_ids_hf, position_ids_hf, labels_hf = build_decoder_inputs_hf()
    outs_hf = hf(
        input_ids=input_ids_hf,
        position_ids=position_ids_hf,
        labels=labels_hf,
    )
    outs_hf.loss.backward()

    input_ids_my, position_ids_my, labels_my = build_decoder_inputs_my()
    input_ids_my_dp = _shard_virtual_dp(input_ids_my, dist_ctx)
    position_ids_my_dp = _shard_virtual_dp(position_ids_my, dist_ctx)
    labels_my_dp = _shard_virtual_dp(labels_my, dist_ctx)

    loss_scale_step_size = (labels_my != -100).sum().item()

    my_loss_accum = []

    def _loss_fn(outputs, mb_index):
        # weight = (labels_my_dp != -100).sum(dim=-1)[mb_index]
        ret = outputs["logps"].sum() / loss_scale_step_size
        my_loss_accum.append(ret.detach())
        return ret

    schedule, _ = build_schedule(
        dist_ctx,
        n_microbatches=2,
        schedule_config=PipelineScheduleLoopedBFSConfig(
            num_stages_per_rank=2
        ),
        model_provider=lambda stg: _build_distributed_model(
            stage=stg,
            hf=hf,
            dist_ctx=dist_ctx,
            stage_memo=stage_memo,
            checkpointing=checkpointing
        ),
        loss_fn=_loss_fn,
        sharding_spec=PipelineShardingSpec()
    )

    inputs_my_dp = {
        "input_ids": input_ids_my_dp
    }
    kwargs_my_dp = {
        "position_ids": position_ids_my_dp,
        "labels": labels_my_dp
    }

    schedule.schedule.configure_buffers(inputs_my_dp, kwargs_my_dp)

    schedule.schedule.step(
        inputs_my_dp,
        kwargs_my_dp
    )

    if dense_mesh.get_local_rank("pp") == 1:
        loss_my = DTensor.from_local(
            torch.tensor(my_loss_accum).sum(),
            device_mesh=dense_mesh["dp_replicate"],
            placements=(Partial("sum"),)
        ).full_tensor()
        assert torch.allclose(outs_hf.loss, loss_my, atol=1e-4, rtol=0.001)

    for stage, stage_model in stage_memo:
        check_lm_model_grad(my=stage_model, hf=hf, is_dist=True, stage=stage)
