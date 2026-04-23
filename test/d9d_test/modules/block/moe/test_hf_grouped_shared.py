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
from d9d.module.block.moe import MoELayer, SharedExpertParameters
from torch import nn
from transformers import Qwen3_5MoeConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock

from d9d_test.modules.block.moe.batch import MOE_HIDDEN_SIZE, build_moe_inputs, materialize_moe_inputs
from d9d_test.modules.helper import (
    assert_mapped_gradients_close,
    clone_module_weights,
    torch_seed,
)

_NUM_EXPERTS = 32
_NUM_ACTIVATE_EXPERTS = 4
_MOE_INTERMEDIATE_SIZE = 256
_SHARED_EXPERT_INTERMEDIATE_SIZE = 1024


def _mapper_from_hf_to_d9d_qwen3_5() -> ModelStateMapper:
    return ModelStateMapperParallel(
        [
            # Grouped experts mapping
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
            # Router mapping
            ModelStateMapperRename("gate.weight", "router.gate.weight"),
            # Shared expert mapping
            ModelStateMapperRename("shared_expert.gate_proj.weight", "shared_expert.expert.gate_proj.weight"),
            ModelStateMapperRename("shared_expert.up_proj.weight", "shared_expert.expert.up_proj.weight"),
            ModelStateMapperRename("shared_expert.down_proj.weight", "shared_expert.expert.down_proj.weight"),
            ModelStateMapperRename("shared_expert_gate.weight", "shared_expert.gate.weight"),
        ]
    )


def build_d9d_moe_shared(dtype: torch.dtype) -> MoELayer:
    with torch_seed(321):
        moe = (
            MoELayer(
                hidden_dim=MOE_HIDDEN_SIZE,
                num_grouped_experts=_NUM_EXPERTS,
                intermediate_dim_grouped=_MOE_INTERMEDIATE_SIZE,
                top_k=_NUM_ACTIVATE_EXPERTS,
                router_renormalize_probabilities=True,
                shared_expert=SharedExpertParameters(
                    intermediate_size=_SHARED_EXPERT_INTERMEDIATE_SIZE,
                    enable_gate=True,
                ),
            )
            .cuda()
            .to(dtype)
        )
        moe.reset_parameters()
    return moe


def build_hf_moe_shared(dtype: torch.dtype) -> Qwen3_5MoeSparseMoeBlock:
    with torch_seed(1312):
        # Build the HF layer
        module = (
            Qwen3_5MoeSparseMoeBlock(
                Qwen3_5MoeConfig(
                    num_experts=_NUM_EXPERTS,
                    num_experts_per_tok=_NUM_ACTIVATE_EXPERTS,
                    hidden_size=MOE_HIDDEN_SIZE,
                    moe_intermediate_size=_MOE_INTERMEDIATE_SIZE,
                    shared_expert_intermediate_size=_SHARED_EXPERT_INTERMEDIATE_SIZE,
                    hidden_act="silu",
                )
            )
            .cuda()
            .to(dtype)
        )

        # Initialize Sparse Experts
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

        # Initialize Shared Expert
        nn.init.uniform_(
            module.shared_expert.gate_proj.weight,
            -1 / math.sqrt(MOE_HIDDEN_SIZE),
            1 / math.sqrt(MOE_HIDDEN_SIZE),
        )
        nn.init.uniform_(
            module.shared_expert.up_proj.weight,
            -1 / math.sqrt(MOE_HIDDEN_SIZE),
            1 / math.sqrt(MOE_HIDDEN_SIZE),
        )
        nn.init.uniform_(
            module.shared_expert.down_proj.weight,
            -1 / math.sqrt(_SHARED_EXPERT_INTERMEDIATE_SIZE),
            1 / math.sqrt(_SHARED_EXPERT_INTERMEDIATE_SIZE),
        )
        # Initialize the shared expert's gate block
        nn.init.uniform_(
            module.shared_expert_gate.weight,
            -1 / math.sqrt(MOE_HIDDEN_SIZE),
            1 / math.sqrt(MOE_HIDDEN_SIZE),
        )

        nn.init.uniform_(
            module.gate.weight,
            -1 / math.sqrt(_NUM_EXPERTS),
            1 / math.sqrt(_NUM_EXPERTS),
        )

        return module


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf_qwen3_5_shared(dtype: torch.dtype):
    mapper = _mapper_from_hf_to_d9d_qwen3_5()

    init = build_moe_inputs(dtype)

    # Run HF module
    inputs_hf = materialize_moe_inputs(init)
    module_hf = build_hf_moe_shared(dtype)
    out_hf = module_hf(inputs_hf.hidden_states + inputs_hf.pre)
    out_hf.mean().backward()

    # Run d9d module
    inputs_d9d = materialize_moe_inputs(init)
    module_d9d = build_d9d_moe_shared(dtype)
    clone_module_weights(from_module=module_hf, to_module=module_d9d, map_with=mapper)
    out_d9d = module_d9d(inputs_d9d.hidden_states + inputs_d9d.pre)
    out_d9d.mean().backward()

    # Check Tolerance and equivalence
    torch.testing.assert_close(out_d9d, out_hf, atol=3e-3, rtol=1e-2)
    torch.testing.assert_close(out_d9d.mean(), out_hf.mean(), atol=1e-3, rtol=1e-2)
    torch.testing.assert_close(inputs_d9d.pre.grad, inputs_hf.pre.grad, atol=1e-7, rtol=0.01)

    # Assert gradient matching for all mapped weights (including the newly added shared expert ones)
    assert_mapped_gradients_close(from_module=module_hf, to_module=module_d9d, map_with=mapper)
