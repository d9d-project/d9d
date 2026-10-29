import pytest
import torch
from d9d.core.dist_context import BATCH_DOMAIN, EXPERT_DOMAIN, DeviceMeshParameters
from d9d.module.block.moe import MoELayer, SharedExpertParameters
from d9d.module.parallelism.api import parallelize_expert_parallel

from d9d_test.modules.block.moe.batch import MOE_HIDDEN_SIZE, build_moe_inputs, materialize_moe_inputs
from d9d_test.modules.helper import (
    check_grad_distance_all_local_dist,
    copy_params_local_to_dist,
    sync_grads_manually,
    torch_seed,
)

_NUM_EXPERTS = 32
_NUM_ACTIVATE_EXPERTS = 4
_MOE_INTERMEDIATE_SIZE = 256
_MOE_INTERMEDIATE_SIZE_SHARED = 384


def build_d9d_moe(dtype: torch.dtype, shared_expert_params: SharedExpertParameters | None) -> MoELayer:
    with torch_seed(42):
        moe = (
            MoELayer(
                hidden_dim=MOE_HIDDEN_SIZE,
                num_grouped_experts=_NUM_EXPERTS,
                intermediate_dim_grouped=_MOE_INTERMEDIATE_SIZE,
                top_k=_NUM_ACTIVATE_EXPERTS,
                router_renormalize_probabilities=True,
                shared_expert=shared_expert_params,
            )
            .cuda()
            .to(dtype)
        )
        moe.reset_parameters()
    return moe


@pytest.mark.distributed
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "shared_expert_params",
    [
        None,
        SharedExpertParameters(intermediate_size=_MOE_INTERMEDIATE_SIZE_SHARED, enable_gate=True),
        SharedExpertParameters(intermediate_size=_MOE_INTERMEDIATE_SIZE_SHARED, enable_gate=False),
    ],
)
def test_consistent_to_itself_expert_parallel(dtype: torch.dtype, dist_ctx_factory, shared_expert_params):
    ctx = dist_ctx_factory(DeviceMeshParameters(expert_parallel=8, data_parallel_replicate=8))
    init = build_moe_inputs(dtype)

    # Run Local
    local_inputs = materialize_moe_inputs(init)
    local = build_d9d_moe(dtype, shared_expert_params)
    out_local = local(local_inputs.hidden_states + local_inputs.pre)
    loss_local = out_local.sum() / local_inputs.hidden_states.shape[-1]
    loss_local.backward()

    # Run dist
    dist_inputs = materialize_moe_inputs(init)
    dist_model = build_d9d_moe(dtype, shared_expert_params)
    parallelize_expert_parallel(
        dist_model,
        mesh_experts=ctx.mesh_for(EXPERT_DOMAIN)["ep_replicate", "ep_shard"],
    )
    copy_params_local_to_dist(local, dist_model)
    out_dist = dist_model(dist_inputs.hidden_states + dist_inputs.pre)
    dp_size = int(ctx.mesh_for(BATCH_DOMAIN)["dp"].size())
    loss_dist = (out_dist.sum() / dist_inputs.hidden_states.shape[-1]) / dp_size
    loss_dist.backward()

    # Check
    torch.testing.assert_close(out_dist, out_local, atol=2e-3, rtol=1e-2)

    sync_grads_manually(dist_model)
    check_grad_distance_all_local_dist(local, dist_model)

    torch.testing.assert_close(dist_inputs.pre.grad * dp_size, local_inputs.pre.grad, atol=1e-3, rtol=0.01)
