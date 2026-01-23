import pytest
import torch
from d9d.core.dist_context import BATCH_DOMAIN, EXPERT_DOMAIN, DeviceMeshParameters
from d9d.module.block.moe import MoELayer
from d9d.module.parallelism.api import parallelize_expert_parallel
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from d9d_test.modules.block.moe.util import check_moe_qwen3_moe_grad, clone_moe_weights_qwen3_moe
from d9d_test.modules.checkers import check_grad

_NUM_DEVICES = 8


def build_moe_inputs(dtype: torch.dtype):
    torch.manual_seed(4242)

    hidden_states = torch.randn(16, 1024, 512).cuda().to(dtype)
    moe_hf_pre = nn.Parameter(torch.zeros((1, 1, 512), dtype=dtype, device="cuda"))
    moe_my_pre = nn.Parameter(torch.zeros((1, 1, 512), dtype=dtype, device="cuda"))

    return hidden_states, moe_hf_pre, moe_my_pre


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

    moe_my = MoELayer(
        hidden_dim=512,
        num_grouped_experts=32,
        intermediate_dim_grouped=256,
        top_k=4,
        router_renormalize_probabilities=True
    ).cuda().to(dtype)

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

    check_grad(moe_my_pre.grad, moe_hf_pre.grad, is_dist=False, atol=1e-7, rtol=0.01)
    check_moe_qwen3_moe_grad(moe_my, moe_hf, is_dist=False)


@pytest.mark.distributed
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf_expert_parallel(dtype):
    ctx = DeviceMeshParameters(
        pipeline_parallel=1,
        tensor_parallel=1,
        expert_parallel=8,
        data_parallel_shard=1,
        context_parallel_shard=1,
        data_parallel_replicate=8,
        context_parallel_replicate=1
    ).build()
    expert_mesh = ctx.mesh_for(EXPERT_DOMAIN)
    batch_mesh = ctx.mesh_for(BATCH_DOMAIN)

    hidden_states_in, moe_hf_pre, moe_my_pre = build_moe_inputs(dtype)
    moe_hf, moe_my = build_hf_my_moe(dtype)

    hidden_states_hf, _ = moe_hf(hidden_states_in + moe_hf_pre)
    hidden_states_hf.mean().backward()

    parallelize_expert_parallel(
        moe_my,
        mesh_experts=expert_mesh[["ep_replicate", "ep_shard"]]
    )
    moe_my_pre_dist = distribute_tensor(
        moe_my_pre,
        device_mesh=batch_mesh["dp"],
        placements=[Replicate()]
    )

    hidden_states_dp = DTensor.from_local(
        hidden_states_in,
        device_mesh=batch_mesh["dp"],
        placements=[Replicate()]
    ).redistribute(placements=(Shard(0),))

    assert hidden_states_dp.to_local().shape == (16 // _NUM_DEVICES, 1024, 512)
    hidden_states_my = moe_my((hidden_states_dp + moe_my_pre_dist).to_local())

    (hidden_states_my.mean() / batch_mesh["dp"].size()).backward()
    hidden_states_my = DTensor.from_local(
        hidden_states_my,
        device_mesh=batch_mesh["dp"],
        placements=[Shard(0)]
    ).redistribute(placements=(Replicate(),)).to_local()

    assert torch.allclose(hidden_states_my, hidden_states_hf, atol=1e-3, rtol=0.01)

    check_grad(moe_my_pre_dist.grad, moe_hf_pre.grad, is_dist=True, atol=1e-7, rtol=0.01)
    check_moe_qwen3_moe_grad(moe_my, moe_hf, is_dist=True)
