import dataclasses
import math

import pytest
import torch
from d9d.core.dist_context import BATCH_DOMAIN, EXPERT_DOMAIN, DeviceMeshParameters
from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperSequential
from d9d.model_state.mapper.leaf import (
    ModelStateMapperChunkTensors,
    ModelStateMapperRename,
    ModelStateMapperTranspose,
)
from d9d.module.block.moe import MoELayer
from d9d.module.parallelism.api import parallelize_expert_parallel
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from d9d_test.modules.helper import (
    assert_mapped_gradients_close,
    check_grad_distance_all_local_dist,
    clone_module_weights,
    copy_params_local_to_dist,
    forward_tolerance_for,
    sync_grads_manually,
    torch_seed,
)

_NUM_EXPERTS = 32
_NUM_ACTIVATE_EXPERTS = 4
_HIDDEN_SIZE = 512
_MOE_INTERMEDIATE_SIZE = 256


def _mapper_from_hf_to_d9d() -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            ModelStateMapperSequential(
                [
                    ModelStateMapperTranspose("experts.gate_up_proj", dims=(-1, -2)),
                    ModelStateMapperChunkTensors(
                        source_name="experts.gate_up_proj",
                        target_names=[
                            "grouped_experts.gate_proj.weight",
                            "grouped_experts.up_proj.weight",
                        ],
                        dim=-1,
                    ),
                ]
            ),
            ModelStateMapperSequential(
                [
                    ModelStateMapperTranspose("experts.down_proj", dims=(-1, -2)),
                    ModelStateMapperRename("experts.down_proj", "grouped_experts.down_proj.weight"),
                ]
            ),
            ModelStateMapperRename("gate.weight", "router.gate.weight"),
        ]
    )


@dataclasses.dataclass(frozen=True)
class MoEInputsInit:
    hidden_states: torch.Tensor
    pre_init: torch.Tensor


@dataclasses.dataclass(frozen=True)
class MoEInputs:
    hidden_states: torch.Tensor
    pre: torch.nn.Parameter


def build_moe_inputs(dtype: torch.dtype) -> MoEInputsInit:
    with torch_seed(4242):
        return MoEInputsInit(
            hidden_states=torch.randn((16, 1024, _HIDDEN_SIZE), device="cuda", dtype=dtype),
            pre_init=torch.zeros((1, 1, _HIDDEN_SIZE), device="cuda", dtype=dtype),
        )


def materialize_moe_inputs(init: MoEInputsInit) -> MoEInputs:
    return MoEInputs(
        hidden_states=init.hidden_states.clone(),
        pre=torch.nn.Parameter(init.pre_init.clone()),
    )


def build_d9d_moe(dtype: torch.dtype) -> MoELayer:
    with torch_seed(42):
        moe = (
            MoELayer(
                hidden_dim=_HIDDEN_SIZE,
                num_grouped_experts=_NUM_EXPERTS,
                intermediate_dim_grouped=_MOE_INTERMEDIATE_SIZE,
                top_k=_NUM_ACTIVATE_EXPERTS,
                router_renormalize_probabilities=True,
            )
            .cuda()
            .to(dtype)
        )
        moe.reset_parameters()
    return moe


def build_hf_moe(dtype: torch.dtype) -> Qwen3MoeSparseMoeBlock:
    with torch_seed(43):
        module = (
            Qwen3MoeSparseMoeBlock(
                Qwen3MoeConfig(
                    num_experts=_NUM_EXPERTS,
                    num_experts_per_tok=_NUM_ACTIVATE_EXPERTS,
                    norm_topk_prob=True,
                    hidden_size=_HIDDEN_SIZE,
                    moe_intermediate_size=_MOE_INTERMEDIATE_SIZE,
                    hidden_act="silu",
                )
            )
            .cuda()
            .to(dtype)
        )
        nn.init.uniform_(
            module.experts.gate_up_proj,
            -1 / math.sqrt(_HIDDEN_SIZE),
            1 / math.sqrt(_HIDDEN_SIZE),
        )
        nn.init.uniform_(
            module.experts.down_proj,
            -1 / math.sqrt(_MOE_INTERMEDIATE_SIZE),
            1 / math.sqrt(_MOE_INTERMEDIATE_SIZE),
        )
        return module


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf(dtype: torch.dtype):
    mapper = _mapper_from_hf_to_d9d()

    init = build_moe_inputs(dtype)

    # Run HF module
    inputs_hf = materialize_moe_inputs(init)
    module_hf = build_hf_moe(dtype)
    out_hf = module_hf(inputs_hf.hidden_states + inputs_hf.pre)
    out_hf.mean().backward()

    # Run d9d module
    inputs_d9d = materialize_moe_inputs(init)
    module_d9d = build_d9d_moe(dtype)
    clone_module_weights(from_module=module_hf, to_module=module_d9d, map_with=mapper)
    out_d9d = module_d9d(inputs_d9d.hidden_states + inputs_d9d.pre)
    out_d9d.mean().backward()

    # Check
    tol = forward_tolerance_for(dtype)
    torch.testing.assert_close(
        out_d9d,
        out_hf,
        atol=tol.atol,
        rtol=tol.rtol,
    )
    torch.testing.assert_close(
        out_d9d.mean(),
        out_hf.mean(),
        atol=tol.atol,
        rtol=tol.rtol,
    )
    torch.testing.assert_close(
        inputs_d9d.pre.grad,
        inputs_hf.pre.grad,
        atol=1e-7,
        rtol=0.01,
    )
    assert_mapped_gradients_close(from_module=module_hf, to_module=module_d9d, map_with=mapper)


@pytest.mark.distributed
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_itself_expert_parallel(dtype: torch.dtype, dist_ctx_factory):
    ctx = dist_ctx_factory(DeviceMeshParameters(expert_parallel=8, data_parallel_replicate=8))
    init = build_moe_inputs(dtype)

    # Run Local
    local_inputs = materialize_moe_inputs(init)
    local = build_d9d_moe(dtype)
    out_local = local(local_inputs.hidden_states + local_inputs.pre)
    loss_local = out_local.sum() / local_inputs.hidden_states.shape[-1]
    loss_local.backward()

    # Run dist
    dist_inputs = materialize_moe_inputs(init)
    dist_model = build_d9d_moe(dtype)
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
    tol = forward_tolerance_for(dtype)
    torch.testing.assert_close(
        out_dist,
        out_local,
        atol=tol.atol,
        rtol=tol.rtol,
    )

    sync_grads_manually(dist_model)
    check_grad_distance_all_local_dist(local, dist_model)

    torch.testing.assert_close(
        dist_inputs.pre.grad * dp_size,
        local_inputs.pre.grad,
        atol=1e-3,
        rtol=0.01,
    )
