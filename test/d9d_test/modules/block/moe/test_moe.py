import pytest
import torch
from d9d.core.dist_context import BATCH_DOMAIN, EXPERT_DOMAIN, DeviceMeshParameters
from d9d.module.block.moe import MoELayer
from d9d.module.parallelism.api import parallelize_expert_parallel
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from d9d_test.modules.block.moe.util import check_moe_qwen3_moe_grad, clone_moe_weights_qwen3_moe
from d9d_test.modules.checkers import check_grad, check_grad_distance_all_local_dist
from d9d_test.modules.grad_sync import sync_grads_manually

_NUM_DEVICES = 8


def build_moe_inputs(dtype: torch.dtype):
    torch.manual_seed(4242)

    hidden_states = torch.randn(16, 1024, 512).cuda().to(dtype)
    moe_hf_pre = nn.Parameter(torch.zeros((1, 1, 512), dtype=dtype, device="cuda"))
    moe_my_pre = nn.Parameter(torch.zeros((1, 1, 512), dtype=dtype, device="cuda"))

    return hidden_states, moe_hf_pre, moe_my_pre


def build_my_moe(dtype: torch.dtype):
    torch.manual_seed(42)

    moe = MoELayer(
        hidden_dim=512,
        num_grouped_experts=32,
        intermediate_dim_grouped=256,
        top_k=4,
        router_renormalize_probabilities=True
    ).cuda().to(dtype)
    moe.reset_parameters()
    return moe


def build_hf_my_moe(dtype: torch.dtype):
    torch.manual_seed(42)

    moe_hf = Qwen3MoeSparseMoeBlock(
        Qwen3MoeConfig(
            num_experts=32,
            num_experts_per_tok=4,
            norm_topk_prob=True,
            hidden_size=512,
            moe_intermediate_size=256,
            hidden_act="silu"
        )
    ).cuda().to(dtype)

    moe_my = build_my_moe(dtype)

    clone_moe_weights_qwen3_moe(my=moe_my, hf=moe_hf)

    return moe_hf, moe_my


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf(dtype):
    hidden_states, moe_hf_pre, moe_my_pre = build_moe_inputs(dtype)
    moe_hf, moe_my = build_hf_my_moe(dtype)

    hidden_states_hf, _ = moe_hf(hidden_states + moe_hf_pre)
    hidden_states_my = moe_my(hidden_states + moe_my_pre)

    assert torch.allclose(hidden_states_my, hidden_states_hf, atol=1e-3, rtol=0.01)

    hidden_states_hf.mean().backward()
    hidden_states_my.mean().backward()

    check_grad(moe_my_pre.grad, moe_hf_pre.grad, atol=1e-7, rtol=0.01)
    check_moe_qwen3_moe_grad(moe_my, moe_hf)


@pytest.mark.distributed
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_itself_expert_parallel(dtype, dist_ctx_factory):
    ctx = dist_ctx_factory(DeviceMeshParameters(
        expert_parallel=8,
        data_parallel_replicate=8
    ))
    expert_mesh = ctx.mesh_for(EXPERT_DOMAIN)
    batch_mesh = ctx.mesh_for(BATCH_DOMAIN)
    dp_size = batch_mesh["dp"].size()

    hidden_states_in, pre_local, pre_dist = build_moe_inputs(dtype)

    loss_div_factor = hidden_states_in.shape[-1]

    moe_local = build_my_moe(dtype)
    hidden_states_local = moe_local(hidden_states_in + pre_local)
    (hidden_states_local.sum() / loss_div_factor).backward()

    moe_dist = build_my_moe(dtype)
    parallelize_expert_parallel(
        moe_dist,
        mesh_experts=expert_mesh[["ep_replicate", "ep_shard"]]
    )
    hidden_states_dist = moe_dist(hidden_states_in + pre_dist)
    (hidden_states_dist.sum() / loss_div_factor / dp_size).backward()

    sync_grads_manually(moe_dist)

    assert torch.allclose(hidden_states_dist, hidden_states_local, atol=1e-3, rtol=0.01)
    check_grad(pre_dist.grad * 8, pre_local.grad, atol=1e-3, rtol=0.01)
    check_grad_distance_all_local_dist(local=moe_local, dist=moe_dist)
