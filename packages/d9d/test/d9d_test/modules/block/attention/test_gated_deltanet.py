import dataclasses

import pytest
import torch
from d9d.model_state.mapper.compose import ModelStateMapperParallel, ModelStateMapperSequential
from d9d.model_state.mapper.leaf import ModelStateMapperRename, ModelStateMapperSqueeze
from d9d.module.block.attention.linear import GatedDeltaNet, MambaDecayGateParameters
from torch import nn
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeGatedDeltaNet

from d9d_test.modules.helper import assert_mapped_gradients_close, clone_module_weights, torch_seed

_HIDDEN = 512
_HEAD_QK_DIM = 64
_HEAD_V_DIM = 64
_CONV_SIZE = 4
_BATCH = 2
_SEQ = 128
_NORM_EPS = 1e-5
_GATE_LOGIT_NORMALIZER = 16.0


def _build_mapper() -> ModelStateMapperParallel:
    return ModelStateMapperParallel(
        [
            ModelStateMapperSequential(
                [
                    ModelStateMapperRename("conv1d.weight", "qkv_conv1d.weight"),
                    ModelStateMapperSqueeze("qkv_conv1d.weight", dim=1),
                ]
            ),
            ModelStateMapperRename("in_proj_qkv.weight", "qkv_proj.weight"),
            ModelStateMapperRename("in_proj_z.weight", "g_proj.weight"),
            ModelStateMapperRename("in_proj_b.weight", "b_proj.weight"),
            ModelStateMapperRename("in_proj_a.weight", "decay_gate.proj.weight"),
            ModelStateMapperRename("A_log", "decay_gate.A_log"),
            ModelStateMapperRename("dt_bias", "decay_gate.dt_bias"),
            ModelStateMapperRename("norm.weight", "out_norm.weight"),
            ModelStateMapperRename("out_proj.weight", "o_proj.weight"),
        ]
    )


@dataclasses.dataclass
class GdnInputsInit:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    pre_init: torch.Tensor


@dataclasses.dataclass
class GdnInputs:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    pre: nn.Parameter


def build_inputs(dtype: torch.dtype, mask_style: str = "ones") -> GdnInputsInit:
    with torch_seed(4242):
        hidden_states = torch.randn(_BATCH, _SEQ, _HIDDEN).cuda().to(dtype)

        match mask_style:
            case "ones":
                attention_mask = torch.ones(_BATCH, _SEQ).cuda().to(dtype)
            case "padded":
                attention_mask = torch.ones(_BATCH, _SEQ).cuda().to(dtype)
                attention_mask[:, -16:] = 0.0
            case "none":
                attention_mask = None
            case _:
                raise ValueError(f"Unknown mask_style: {mask_style}")

        pre_init = torch.zeros((1, 1, _HIDDEN), device="cuda", dtype=dtype)

        return GdnInputsInit(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            pre_init=pre_init,
        )


def materialize_inputs(init: GdnInputsInit) -> GdnInputs:
    return GdnInputs(
        hidden_states=init.hidden_states.clone(),
        attention_mask=init.attention_mask.clone() if init.attention_mask is not None else None,
        pre=torch.nn.Parameter(init.pre_init.clone()),
    )


def build_hf_qwen3_5_moe(
    dtype: torch.dtype,
    *,
    num_value_heads: int,
    num_key_heads: int,
) -> Qwen3_5MoeGatedDeltaNet:
    with torch_seed(42):
        config = Qwen3_5MoeTextConfig(
            hidden_size=_HIDDEN,
            linear_num_value_heads=num_value_heads,
            linear_num_key_heads=num_key_heads,
            linear_key_head_dim=_HEAD_QK_DIM,
            linear_value_head_dim=_HEAD_V_DIM,
            linear_conv_kernel_dim=_CONV_SIZE,
            rms_norm_eps=_NORM_EPS,
            hidden_act="silu",
        )
        return Qwen3_5MoeGatedDeltaNet(config, layer_idx=0).cuda().to(dtype)


def build_d9d(
    dtype: torch.dtype,
    *,
    num_qk_heads: int,
    num_value_heads: int,
) -> GatedDeltaNet:
    with torch_seed(42):
        return (
            GatedDeltaNet(
                hidden_size=_HIDDEN,
                num_query_key_heads=num_qk_heads,
                num_value_heads=num_value_heads,
                head_qk_dim=_HEAD_QK_DIM,
                head_v_dim=_HEAD_V_DIM,
                conv_size=_CONV_SIZE,
                decay_gate=MambaDecayGateParameters(
                    normalizer=_GATE_LOGIT_NORMALIZER,
                    dt_max=0.1,
                    dt_min=0.001,
                    dt_init_floor=1e-4,
                ),
                norm_eps=_NORM_EPS,
                use_qk_l2norm=True,
            )
            .cuda()
            .to(dtype)
        )


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mask_style", ["ones", "none", "padded"])
@pytest.mark.parametrize(
    ("num_value_heads", "num_key_heads"),
    [
        pytest.param(8, 8, id="matched_kv_heads"),
        pytest.param(16, 4, id="grouped_kv_heads"),
    ],
)
def test_consistent_to_hf(
    dtype: torch.dtype,
    mask_style: str,
    num_value_heads: int,
    num_key_heads: int,
) -> None:
    inputs_init = build_inputs(dtype, mask_style=mask_style)
    mapper = _build_mapper()

    # HF
    inputs_hf = materialize_inputs(inputs_init)
    module_hf = build_hf_qwen3_5_moe(dtype, num_value_heads=num_value_heads, num_key_heads=num_key_heads)
    out_hf = module_hf(
        hidden_states=inputs_hf.hidden_states + inputs_hf.pre,
        attention_mask=inputs_hf.attention_mask,
    )
    out_hf.mean().backward()

    # d9d
    inputs_d9d = materialize_inputs(inputs_init)
    module_d9d = build_d9d(dtype, num_qk_heads=num_key_heads, num_value_heads=num_value_heads)
    clone_module_weights(module_hf, module_d9d, map_with=mapper)

    out_d9d = module_d9d(
        inputs_d9d.hidden_states + inputs_d9d.pre,
        attention_mask=inputs_d9d.attention_mask,
    )
    out_d9d.mean().backward()

    # Check
    torch.testing.assert_close(out_d9d, out_hf, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(inputs_d9d.pre.grad, inputs_hf.pre.grad, atol=1e-6, rtol=0.01)
    assert_mapped_gradients_close(from_module=module_hf, to_module=module_d9d, map_with=mapper)
