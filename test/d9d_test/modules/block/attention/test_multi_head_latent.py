import dataclasses

import pytest
import torch
from d9d.module.block.attention import MultiHeadLatentAttention
from d9d.module.block.positional import RotaryEmbeddingStyle
from torch import nn
from transformers import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention

from d9d_test.modules.block.attention.util import check_mla_deepseek_v2_grad, clone_mla_deepseek_v2
from d9d_test.modules.checkers import check_grad

# ---------------------------------------------------------------------------
# Test configuration
# hidden=512, n_heads=8, nope=32, rope=16, v=32, kv_lora=64, q_lora=192
# full QK head_dim = nope + rope = 48, V head_dim = 32
# ---------------------------------------------------------------------------
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


def build_mla_with_q_lora(dtype: torch.dtype) -> MultiHeadLatentAttention:
    torch.manual_seed(42)
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
            is_causal=True,
            rope_style=RotaryEmbeddingStyle.INTERLEAVED,
        )
        .cuda()
        .to(dtype)
    )


def build_mla_no_q_lora(dtype: torch.dtype) -> MultiHeadLatentAttention:
    torch.manual_seed(42)
    return (
        MultiHeadLatentAttention(
            hidden_size=_HIDDEN,
            num_attention_heads=_N_HEADS,
            qk_nope_head_dim=_NOPE,
            qk_rope_head_dim=_ROPE,
            v_head_dim=_V_DIM,
            kv_lora_rank=_KV_LORA_RANK,
            q_lora_rank=None,
            qk_down_norm_eps=_NORM_EPS,
            is_causal=True,
            rope_style=RotaryEmbeddingStyle.INTERLEAVED,
        )
        .cuda()
        .to(dtype)
    )


def build_inputs(dtype: torch.dtype):
    torch.manual_seed(4242)
    hidden_states = torch.randn(_BATCH, _SEQ, _HIDDEN).cuda().to(dtype)

    angles = torch.randn(_BATCH, _SEQ, _ROPE, dtype=torch.float32, device="cuda")
    rope_cos = angles.cos().to(dtype)
    rope_sin = angles.sin().to(dtype)

    return hidden_states, (rope_cos, rope_sin)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_output_shape_with_q_lora(dtype: torch.dtype):
    mla = build_mla_with_q_lora(dtype)
    mla.reset_parameters()
    hidden_states, position_embeddings = build_inputs(dtype)

    out = mla(hidden_states, attention_mask=None, position_embeddings=position_embeddings)

    assert out.shape == (_BATCH, _SEQ, _HIDDEN), f"Expected ({_BATCH}, {_SEQ}, {_HIDDEN}), got {out.shape}"


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_output_shape_no_q_lora(dtype: torch.dtype):
    mla = build_mla_no_q_lora(dtype)
    mla.reset_parameters()
    hidden_states, position_embeddings = build_inputs(dtype)

    out = mla(hidden_states, attention_mask=None, position_embeddings=position_embeddings)

    assert out.shape == (_BATCH, _SEQ, _HIDDEN), f"Expected ({_BATCH}, {_SEQ}, {_HIDDEN}), got {out.shape}"


# ---------------------------------------------------------------------------
# Numerical correctness vs HF DeepseekV2Attention
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MlaInputs:
    hidden_states: torch.Tensor
    hf_pre_tensor: nn.Parameter
    my_pre_tensor: nn.Parameter
    rope: tuple[torch.Tensor, torch.Tensor]
    freqs_cis: torch.Tensor


def build_hf_inputs(dtype: torch.dtype) -> MlaInputs:
    torch.manual_seed(4242)
    hidden_states = torch.randn(_BATCH, _SEQ, _HIDDEN).cuda().to(dtype)
    hf_pre = nn.Parameter(torch.zeros(1, 1, _HIDDEN, dtype=dtype, device="cuda"))
    my_pre = nn.Parameter(torch.zeros(1, 1, _HIDDEN, dtype=dtype, device="cuda"))

    # Identity position embeddings: RoPE has no effect
    # d9d: cos=1, sin=0  →  q_rotated = q*1 + rotate_half(q)*0 = q

    # Generate random rotational phases
    angles = torch.randn(_BATCH, _SEQ, _ROPE // 2, dtype=torch.float32, device="cuda")

    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    rope_cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(dtype)
    rope_sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(dtype)

    return MlaInputs(
        hidden_states=hidden_states,
        hf_pre_tensor=hf_pre,
        my_pre_tensor=my_pre,
        rope=(rope_cos, rope_sin),
        freqs_cis=freqs_cis,
    )


def build_hf_deepseekv2(dtype: torch.dtype):
    torch.manual_seed(42)
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


def build_mla_hf_compat(dtype: torch.dtype) -> MultiHeadLatentAttention:
    """Build MLA with is_causal=False to match HF eager mode (no causal mask when mask=None)."""
    torch.manual_seed(42)
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


def build_hf_my_deepseekv2(dtype: torch.dtype):
    hf = build_hf_deepseekv2(dtype)
    my = build_mla_hf_compat(dtype)
    my.reset_parameters()
    clone_mla_deepseek_v2(my=my, hf=hf)
    return hf, my


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf(dtype: torch.dtype):
    inputs = build_hf_inputs(dtype)
    hf, my = build_hf_my_deepseekv2(dtype)

    out_hf, _ = hf(
        inputs.hidden_states + inputs.hf_pre_tensor,
        attention_mask=None,
        position_embeddings=inputs.freqs_cis,
    )
    out_my = my(
        inputs.hidden_states + inputs.my_pre_tensor,
        attention_mask=None,
        position_embeddings=inputs.rope,
    )

    assert torch.allclose(out_my, out_hf, atol=1e-2, rtol=1e-2), (
        f"Forward output mismatch. Max diff: {(out_my - out_hf).abs().max():.4f}"
    )

    out_hf.mean().backward()
    out_my.mean().backward()

    check_grad(inputs.my_pre_tensor.grad, inputs.hf_pre_tensor.grad, atol=1e-6, rtol=0.01)
    check_mla_deepseek_v2_grad(my, hf)
