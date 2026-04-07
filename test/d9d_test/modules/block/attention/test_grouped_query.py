import dataclasses

import pytest
import torch
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperIdentity
from d9d.module.block.attention import GroupedQueryAttention
from d9d.module.block.positional import RotaryEmbeddingStyle
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from d9d_test.modules.helper import assert_mapped_gradients_close, clone_module_weights, torch_seed


@dataclasses.dataclass(frozen=True)
class AttentionInputsInit:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    rope: tuple[torch.Tensor, torch.Tensor]
    pre_init: torch.Tensor


@dataclasses.dataclass(frozen=True)
class AttentionInputs:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    rope: tuple[torch.Tensor, torch.Tensor]
    pre: torch.nn.Parameter


def build_inputs(dtype: torch.dtype) -> AttentionInputsInit:
    with torch_seed(4242):
        hidden_states = torch.randn(2, 1024, 512, device="cuda", dtype=dtype)
        attention_mask = (
            torch.triu(torch.ones((1024, 1024), device="cuda"), diagonal=1)[None, None, :, :]
            * torch.finfo(torch.bfloat16).min
        )
        rope_cos = torch.randn(2, 1024, 512 // 16, device="cuda", dtype=dtype)
        rope_sin = torch.randn(2, 1024, 512 // 16, device="cuda", dtype=dtype)
        return AttentionInputsInit(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rope=(rope_cos, rope_sin),
            pre_init=torch.zeros((1, 1, 512), device="cuda", dtype=dtype),
        )


def materialize_attention_inputs(init: AttentionInputsInit) -> AttentionInputs:
    rope_cos, rope_sin = init.rope
    return AttentionInputs(
        hidden_states=init.hidden_states.clone(),
        attention_mask=init.attention_mask.clone(),
        rope=(rope_cos.clone(), rope_sin.clone()),
        pre=torch.nn.Parameter(init.pre_init.clone()),
    )


def build_hf(dtype: torch.dtype):
    with torch_seed(42):
        return (
            Qwen3MoeAttention(
                Qwen3MoeConfig(
                    hidden_size=512,
                    num_attention_heads=16,
                    num_key_value_heads=4,
                    attention_dropout=0.0,
                    attention_bias=False,
                    rms_norm_eps=1e-5,
                    sliding_window=None,
                    _attn_implementation="eager",
                ),
                layer_idx=0,
            )
            .cuda()
            .to(dtype)
        )


def build_d9d(dtype: torch.dtype):
    with torch_seed(43):
        module = (
            GroupedQueryAttention(
                hidden_size=512,
                num_attention_heads=16,
                num_key_value_heads=4,
                qk_norm_eps=1e-5,
                head_dim=32,
                is_causal=True,
                rope_style=RotaryEmbeddingStyle.HALF,
            )
            .cuda()
            .to(dtype)
        )
        module.reset_parameters()
    return module


def _build_state_mapper():
    return ModelStateMapperParallel(
        [
            ModelStateMapperIdentity(f"{param_name}.weight")
            for param_name in (
                "k_norm",
                "k_proj",
                "q_norm",
                "q_proj",
                "v_proj",
                "o_proj",
            )
        ]
    )


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf(dtype):
    init = build_inputs(dtype)
    mapper = _build_state_mapper()

    # HF
    inputs_hf = materialize_attention_inputs(init)
    module_hf = build_hf(dtype)

    hidden_states_hf, _ = module_hf(
        inputs_hf.hidden_states + inputs_hf.pre,
        attention_mask=inputs_hf.attention_mask,
        position_embeddings=inputs_hf.rope,
    )
    hidden_states_hf.mean().backward()

    # d9d
    inputs_d9d = materialize_attention_inputs(init)
    module_d9d = build_d9d(dtype)
    clone_module_weights(module_hf, module_d9d, map_with=mapper)

    hidden_states_d9d = module_d9d(
        inputs_d9d.hidden_states + inputs_d9d.pre,
        attention_mask=None,
        position_embeddings=inputs_d9d.rope,
    )
    hidden_states_d9d.mean().backward()

    # Check
    torch.testing.assert_close(
        hidden_states_d9d,
        hidden_states_hf,
        atol=1e-2,
        rtol=1e-2,
    )

    torch.testing.assert_close(inputs_d9d.pre.grad, inputs_hf.pre.grad, atol=1e-6, rtol=0.01)
    assert_mapped_gradients_close(from_module=module_hf, to_module=module_d9d, map_with=mapper)


# TODO(max): add context parallel test with new context parallelism API
