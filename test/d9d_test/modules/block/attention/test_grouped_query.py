import dataclasses

import pytest
import torch
from d9d.module.block.attention import GroupedQueryAttention
from torch import nn
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from d9d_test.modules.block.attention.util import (
    check_grouped_query_attention_qwen3_moe_grad,
    clone_grouped_query_attention_qwen3_moe,
)
from d9d_test.modules.checkers import check_grad

_NUM_DEVICES = 8


@dataclasses.dataclass
class AttentionInputs:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    rope: tuple[torch.Tensor, torch.Tensor]
    hf_pre_tensor: torch.Tensor
    my_pre_tensor: torch.Tensor


def build_inputs(dtype: torch.dtype):
    torch.manual_seed(4242)

    hidden_states = torch.randn(2, 1024, 512).cuda().to(dtype)
    attention_mask = torch.triu(
        torch.ones((1024, 1024)), diagonal=1
    ).cuda()[None, None, :, :] * torch.finfo(torch.bfloat16).min
    rope_cos = torch.randn(2, 1024, 512 // 16).cuda().to(dtype)
    rope_sin = torch.randn(2, 1024, 512 // 16).cuda().to(dtype)

    att_hf_pre = nn.Parameter(torch.zeros((1, 1, 512), dtype=dtype, device="cuda"))
    att_my_pre = nn.Parameter(torch.zeros((1, 1, 512), dtype=dtype, device="cuda"))

    return AttentionInputs(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        rope=(rope_cos, rope_sin),
        hf_pre_tensor=att_hf_pre,
        my_pre_tensor=att_my_pre
    )


def build_my(dtype: torch.dtype):
    torch.manual_seed(42)

    attention_my = GroupedQueryAttention(
        hidden_size=512,
        num_attention_heads=16,
        num_key_value_heads=4,
        qk_norm_eps=1e-5,
        head_dim=32,
        is_causal=True
    ).cuda().to(dtype)
    attention_my.reset_parameters()

    return attention_my


def build_hf_my(dtype: torch.dtype):
    torch.manual_seed(42)

    attention_hf = Qwen3MoeAttention(
        Qwen3MoeConfig(
            hidden_size=512,
            num_attention_heads=16,
            num_key_value_heads=4,
            attention_dropout=0.0,
            attention_bias=False,
            rms_norm_eps=1e-5,
            sliding_window=None,
            _attn_implementation="eager"
        ),
        layer_idx=0
    ).cuda().to(dtype)

    attention_my = build_my(dtype)

    clone_grouped_query_attention_qwen3_moe(my=attention_my, hf=attention_hf)

    return attention_hf, attention_my


@pytest.mark.local
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_consistent_to_hf(dtype):
    inputs = build_inputs(dtype)
    hf, my = build_hf_my(dtype)

    hidden_states_hf, _ = hf(
        inputs.hidden_states + inputs.hf_pre_tensor,
        attention_mask=inputs.attention_mask,
        position_embeddings=inputs.rope
    )
    hidden_states_my = my(
        inputs.hidden_states + inputs.my_pre_tensor,
        # we do not pass att mask for torch sdpa
        attention_mask=None,
        position_embeddings=inputs.rope
    )

    assert torch.allclose(hidden_states_my, hidden_states_hf, atol=1e-2, rtol=1e-2)

    hidden_states_hf.mean().backward()
    hidden_states_my.mean().backward()

    check_grad(inputs.my_pre_tensor.grad, inputs.hf_pre_tensor.grad, atol=1e-6, rtol=0.01)

    check_grouped_query_attention_qwen3_moe_grad(my, hf)

# TODO(max): add context parallel test with new context parallelism API
