import dataclasses

import torch

from d9d_test.modules.helper import torch_seed


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
