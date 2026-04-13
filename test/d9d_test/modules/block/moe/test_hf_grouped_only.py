import math

import pytest
import torch
from d9d.model_state.mapper import ModelStateMapper
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperSequential
from d9d.model_state.mapper.leaf import (
    ModelStateMapperChunkTensors,
    ModelStateMapperRename,
    ModelStateMapperTranspose,
)
from d9d.module.block.moe import MoELayer
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from d9d_test.modules.block.moe.batch import MOE_HIDDEN_SIZE, build_moe_inputs, materialize_moe_inputs
from d9d_test.modules.helper import (
    assert_mapped_gradients_close,
    clone_module_weights,
    forward_tolerance_for,
    torch_seed,
)

_NUM_EXPERTS = 32
_NUM_ACTIVATE_EXPERTS = 4
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


def build_d9d_moe(dtype: torch.dtype) -> MoELayer:
    with torch_seed(42):
        moe = (
            MoELayer(
                hidden_dim=MOE_HIDDEN_SIZE,
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
                    hidden_size=MOE_HIDDEN_SIZE,
                    moe_intermediate_size=_MOE_INTERMEDIATE_SIZE,
                    hidden_act="silu",
                )
            )
            .cuda()
            .to(dtype)
        )
        nn.init.uniform_(
            module.experts.gate_up_proj,
            -1 / math.sqrt(MOE_HIDDEN_SIZE),
            1 / math.sqrt(MOE_HIDDEN_SIZE),
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
    torch.testing.assert_close(out_d9d, out_hf, atol=tol.atol, rtol=tol.rtol)
    torch.testing.assert_close(out_d9d.mean(), out_hf.mean(), atol=tol.atol, rtol=tol.rtol)
    torch.testing.assert_close(inputs_d9d.pre.grad, inputs_hf.pre.grad, atol=1e-7, rtol=0.01)
    assert_mapped_gradients_close(from_module=module_hf, to_module=module_d9d, map_with=mapper)
