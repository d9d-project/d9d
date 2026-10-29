import dataclasses

import pytest
import torch
from d9d.model_state.mapper.compose import ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperRename
from d9d.module.block.attention import MultiHeadLatentAttention
from d9d.module.block.positional import RotaryEmbeddingStyle
from torch import nn
from transformers import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention

from d9d_test.modules.helper import assert_mapped_gradients_close, clone_module_weights, torch_seed

_HIDDEN = 512
_N_HEADS = 8
_NOPE = 32
_ROPE = 16
_V_DIM = 32
_KV_LORA_RANK = 64
_Q_LORA_RANK = 192
_BATCH = 2
_SEQ = 64
_NORM_EPS = 1e-6


def _build_mapper():
    return ModelStateMapperParallel(
        [
            ModelStateMapperRename(f"{a}.weight", f"{b}.weight")
            for a, b in (
                ("q_a_proj", "q_proj.down_proj"),
                ("q_a_layernorm", "q_proj.norm"),
                ("q_b_proj", "q_proj.up_proj"),
                ("kv_a_proj_with_mqa", "kv_down_proj"),
                ("kv_a_layernorm", "kv_down_norm"),
                ("kv_b_proj", "kv_up_proj"),
                ("o_proj", "o_proj"),
            )
        ]
    )


# ---------------------------------------------------------------------------
# Numerical correctness vs HF DeepseekV2Attention
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MlaInputsInit:
    hidden_states: torch.Tensor
    pre_init: torch.Tensor
    rope: tuple[torch.Tensor, torch.Tensor]
    freqs_cis: torch.Tensor


@dataclasses.dataclass
class MlaInputs:
    hidden_states: torch.Tensor
    pre: nn.Parameter
    rope: tuple[torch.Tensor, torch.Tensor]
    freqs_cis: torch.Tensor


def build_inputs(dtype: torch.dtype) -> MlaInputsInit:
    with torch_seed(4242):
        hidden_states = torch.randn(_BATCH, _SEQ, _HIDDEN).cuda().to(dtype)

        # Generate random rotational phases
        angles = torch.randn(_BATCH, _SEQ, _ROPE // 2, dtype=torch.float32, device="cuda")

        freqs_cis = torch.polar(torch.ones_like(angles), angles)

        rope_cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype)
        rope_sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype)

        return MlaInputsInit(
            hidden_states=hidden_states,
            pre_init=torch.zeros((1, 1, 512), device="cuda", dtype=dtype),
            rope=(rope_cos, rope_sin),
            freqs_cis=freqs_cis,
        )


def materialize_inputs(init: MlaInputsInit) -> MlaInputs:
    rope_cos, rope_sin = init.rope
    return MlaInputs(
        hidden_states=init.hidden_states.clone(),
        freqs_cis=init.freqs_cis.clone(),
        rope=(rope_cos.clone(), rope_sin.clone()),
        pre=torch.nn.Parameter(init.pre_init.clone()),
    )


def build_hf_deepseekv2(dtype: torch.dtype):
    with torch_seed(42):
        config = DeepseekV2Config(
            hidden_size=_HIDDEN,
            num_attention_heads=_N_HEADS,
            num_key_value_heads=_N_HEADS,
            qk_nope_head_dim=_NOPE,
            qk_rope_head_dim=_ROPE,
            v_head_dim=_V_DIM,
            kv_lora_rank=_KV_LORA_RANK,
            q_lora_rank=_Q_LORA_RANK,
            rms_norm_eps=_NORM_EPS,
            attention_bias=False,
            _attn_implementation="eager",
            max_position_embeddings=4096,
        )
        return DeepseekV2Attention(config, layer_idx=0).cuda().to(dtype)


def build_d9d(dtype: torch.dtype) -> MultiHeadLatentAttention:
    """Build MLA with is_causal=False to match HF eager mode (no causal mask when mask=None)."""
    with torch_seed(42):
        return (
            MultiHeadLatentAttention(
                hidden_size=_HIDDEN,
                num_attention_heads=_N_HEADS,
                qk_nope_head_dim=_NOPE,
                qk_rope_head_dim=_ROPE,
                v_head_dim=_V_DIM,
                kv_lora_rank=_KV_LORA_RANK,
                q_lora_rank=_Q_LORA_RANK,
                qk_down_norm_eps=_NORM_EPS,
                is_causal=False,
                rope_style=RotaryEmbeddingStyle.INTERLEAVED,
            )
            .cuda()
            .to(dtype)
        )


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf(dtype: torch.dtype):
    inputs_init = build_inputs(dtype)
    mapper = _build_mapper()

    # HF
    inputs_hf = materialize_inputs(inputs_init)
    module_hf = build_hf_deepseekv2(dtype)
    out_hf, _ = module_hf(
        inputs_hf.hidden_states + inputs_hf.pre,
        attention_mask=None,
        position_embeddings=inputs_hf.freqs_cis,
    )
    out_hf.mean().backward()

    # d9d
    inputs_d9d = materialize_inputs(inputs_init)
    module_d9d = build_d9d(dtype)
    clone_module_weights(module_hf, module_d9d, map_with=mapper)
    out_d9d = module_d9d(
        inputs_d9d.hidden_states + inputs_d9d.pre,
        attention_mask=None,
        position_embeddings=inputs_d9d.rope,
    )
    out_d9d.mean().backward()

    # Check
    torch.testing.assert_close(out_d9d, out_hf, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(inputs_d9d.pre.grad, inputs_hf.pre.grad, atol=1e-6, rtol=0.01)
    assert_mapped_gradients_close(from_module=module_hf, to_module=module_d9d, map_with=mapper)
