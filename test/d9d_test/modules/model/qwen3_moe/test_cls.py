import pytest
import torch
import torch.distributed as dist
from d9d.core.dist_context import BATCH_DOMAIN, DENSE_DOMAIN, DeviceMeshParameters, DistributedContext
from d9d.dataset import TokenPoolingType, token_pooling_mask_from_attention_mask
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model.qwen3_moe import (
    Qwen3MoEForClassification,
    Qwen3MoEForClassificationParameters,
    Qwen3MoELayerParameters,
    Qwen3MoEParameters,
)
from d9d.module.parallelism.model.qwen3_moe import (
    parallelize_qwen3_moe_for_classification,
)
from d9d.pipelining.api import PipelineShardingSpec, PipelineStageInfo
from d9d.pipelining.factory import PipelineScheduleGPipeConfig, build_schedule
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3MoeConfig, Qwen3MoeForSequenceClassification

from d9d_test.modules.checkers import check_grad_distance_all_local_dist, copy_params_local_to_dist
from d9d_test.modules.grad_sync import sync_grads_manually
from d9d_test.modules.model.qwen3_moe.util import check_cls_model_grad, clone_cls_model_weights

_PAD_TOKEN_ID = 99


def build_hf_cls_model(num_labels=3) -> Qwen3MoeForSequenceClassification:
    torch.manual_seed(131231)

    hf = Qwen3MoeForSequenceClassification(
        Qwen3MoeConfig(
            vocab_size=100,
            num_hidden_layers=2,
            hidden_size=128,
            moe_intermediate_size=64,
            num_experts=8,
            num_experts_per_tok=7,
            num_attention_heads=4,
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
            _attn_implementation="sdpa",
            num_labels=num_labels,
            pad_token_id=_PAD_TOKEN_ID
        )
    ).cuda()

    hf.model.embed_tokens.bfloat16()
    hf.model.layers.bfloat16()
    hf.model.norm.bfloat16()
    hf.score.bfloat16()
    hf.eval()

    return hf


def build_my_cls_model(stage: PipelineStageInfo, checkpointing: bool, num_labels=3) -> Qwen3MoEForClassification:
    torch.manual_seed(123)

    params = Qwen3MoEForClassificationParameters(
        model=Qwen3MoEParameters(
            layer=Qwen3MoELayerParameters(
                hidden_size=128,
                intermediate_size=64,
                num_experts=8,
                experts_top_k=7,
                num_attention_heads=4,
                num_key_value_heads=4,
                rms_norm_eps=1e-5,
                head_dim=32
            ),
            rope_base=10_000,
            max_position_ids=15000,
            num_hidden_layers=2,
            split_vocab_size={"a": 100},
            split_vocab_order=["a"]
        ),
        num_labels=num_labels,
        classifier_dropout=0.0
    )

    model = Qwen3MoEForClassification(
        params=params,
        stage=stage,
        hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
        enable_checkpointing=checkpointing
    ).cuda().bfloat16()
    model.reset_parameters()
    return model


def build_inputs_cls():
    torch.manual_seed(428)
    batch_size = 8
    seq_len = 32

    input_ids = torch.randint(0, 90, (batch_size, seq_len), device="cuda", dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    input_ids[1, -1:] = _PAD_TOKEN_ID
    attention_mask[1, -1:] = 0

    input_ids[2, -5:] = _PAD_TOKEN_ID
    attention_mask[2, -5:] = 0

    position_ids = torch.arange(seq_len, device="cuda", dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    labels = torch.randint(0, 3, (batch_size,), device="cuda", dtype=torch.long)

    return input_ids, attention_mask, position_ids, labels


@pytest.mark.local
@pytest.mark.parametrize("checkpointing", [True, False])
def test_consistent_to_hf(checkpointing):
    stage_info = PipelineStageInfo(current_stage=0, num_stages=1)
    num_labels = 3

    hf = build_hf_cls_model(num_labels)
    my = build_my_cls_model(stage_info, checkpointing, num_labels)

    clone_cls_model_weights(my=my, hf=hf, stage=stage_info)

    input_ids, attention_mask, position_ids, labels = build_inputs_cls()

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        outs_hf = hf(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels
        )

    pooling_mask = token_pooling_mask_from_attention_mask(attention_mask, TokenPoolingType.last)

    outs_my = my(
        input_ids=input_ids,
        position_ids=position_ids,
        pooling_mask=pooling_mask
    )

    loss_fct = nn.CrossEntropyLoss()
    my_scores = outs_my["scores"]
    # we have to convert scores to bf16 since hf does this 😭
    assert my_scores.dtype == torch.float32
    loss_my = loss_fct(my_scores.bfloat16(), labels)

    assert torch.allclose(outs_hf.loss, loss_my, atol=1e-4, rtol=0.001)
    assert torch.allclose(outs_hf.logits, my_scores.bfloat16(), atol=1e-2, rtol=0.01)

    outs_hf.loss.backward()
    loss_my.backward()

    check_cls_model_grad(my, hf, stage=stage_info)


def _build_distributed_cls_model(
        stage: PipelineStageInfo,
        dist_ctx: DistributedContext,
        local_model: nn.Module,
        stage_memo: list,
        checkpointing: bool
):
    model = build_my_cls_model(stage=stage, checkpointing=checkpointing, num_labels=3)

    parallelize_qwen3_moe_for_classification(dist_ctx, model, stage)

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
])
def test_consistent_to_itself_dist(mesh, checkpointing, dist_ctx_factory):
    dist_ctx = dist_ctx_factory(mesh)
    stage_memo = []

    local_stage = PipelineStageInfo(num_stages=1, current_stage=0)
    local = build_my_cls_model(stage=local_stage, checkpointing=checkpointing, num_labels=3)

    input_ids, attention_mask, position_ids, labels = build_inputs_cls()
    loss_div = labels.numel()

    pooling_mask = token_pooling_mask_from_attention_mask(attention_mask, TokenPoolingType.last)

    outs_local = local(
        input_ids=input_ids,
        position_ids=position_ids,
        pooling_mask=pooling_mask
    )

    loss_fct = nn.CrossEntropyLoss()

    loss_local = loss_fct(outs_local["scores"], labels)
    loss_local.backward()

    input_ids_dist = _shard_virtual_dp(input_ids, dist_ctx)
    position_ids_dist = _shard_virtual_dp(position_ids, dist_ctx)
    pooling_mask_dist = _shard_virtual_dp(pooling_mask, dist_ctx)  # Shard Mask same as input
    labels_dist = _shard_virtual_dp(labels, dist_ctx)  # Shard Labels (batch dim)

    loss_dist_accum = []

    def _loss_fn(outputs, microbatch_idx):
        target = labels_dist[microbatch_idx, None]
        loss_val = loss_fct(outputs["scores"], target) / loss_div
        loss_dist_accum.append(loss_val.detach())
        return loss_val

    schedule, _ = build_schedule(
        dist_ctx,
        n_microbatches=2,
        schedule_config=PipelineScheduleGPipeConfig(),
        model_provider=lambda stg: _build_distributed_cls_model(
            stage=stg,
            dist_ctx=dist_ctx,
            stage_memo=stage_memo,
            local_model=local,
            checkpointing=checkpointing
        ),
        callback=_loss_fn
    )

    inputs_dist = {"input_ids": input_ids_dist}
    kwargs_dist = {
        "position_ids": position_ids_dist,
        "pooling_mask": pooling_mask_dist
    }

    schedule.schedule.configure_buffers(inputs_dist, kwargs_dist, sharding_spec=PipelineShardingSpec())

    schedule.schedule.step(inputs_dist, kwargs_dist)

    pp_mesh = dist_ctx.mesh_for(DENSE_DOMAIN)["pp"]

    if pp_mesh.get_local_rank() == pp_mesh.size() - 1:
        loss_dist = torch.stack(loss_dist_accum).sum()

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
