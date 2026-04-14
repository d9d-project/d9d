import dataclasses

import torch
from d9d.module.block.positional import RotaryEmbeddingStyle
from d9d.module.block.positional.rope import prepare_rotary_cos_sin_emb

from d9d_test.modules.helper import torch_seed

BATCH = 2
SEQ_LEN = 1024
HIDDEN_SIZE = 512


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


def build_inputs(dtype: torch.dtype, rope_dim: int = HIDDEN_SIZE // 16) -> AttentionInputsInit:
    with torch_seed(4242):
        hidden_states = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE, device="cuda", dtype=dtype)
        attention_mask = (
            torch.triu(torch.ones((SEQ_LEN, SEQ_LEN), device="cuda"), diagonal=1)[None, None, :, :]
            * torch.finfo(torch.bfloat16).min
        )
        cos_emb, sin_emb = prepare_rotary_cos_sin_emb(
            rope_base=10000,
            head_dim=rope_dim,
            max_position_ids=SEQ_LEN,
            device=torch.device("cuda"),
            dtype=dtype,
            style=RotaryEmbeddingStyle.HALF,
        )
        rope_cos = cos_emb.unsqueeze(0).expand(BATCH, -1, -1)
        rope_sin = sin_emb.unsqueeze(0).expand(BATCH, -1, -1)
        return AttentionInputsInit(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rope=(rope_cos, rope_sin),
            pre_init=torch.zeros((1, 1, HIDDEN_SIZE), device="cuda", dtype=dtype),
        )


def materialize_attention_inputs(init: AttentionInputsInit) -> AttentionInputs:
    rope_cos, rope_sin = init.rope
    return AttentionInputs(
        hidden_states=init.hidden_states.clone(),
        attention_mask=init.attention_mask.clone(),
        rope=(rope_cos.clone(), rope_sin.clone()),
        pre=torch.nn.Parameter(init.pre_init.clone()),
    )
