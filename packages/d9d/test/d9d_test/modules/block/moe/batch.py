import dataclasses

import torch

from d9d_test.modules.helper import (
    torch_seed,
)

MOE_HIDDEN_SIZE = 512


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
            hidden_states=torch.randn((16, 1024, MOE_HIDDEN_SIZE), device="cuda", dtype=dtype),
            pre_init=torch.zeros((1, 1, MOE_HIDDEN_SIZE), device="cuda", dtype=dtype),
        )


def materialize_moe_inputs(init: MoEInputsInit) -> MoEInputs:
    return MoEInputs(
        hidden_states=init.hidden_states.clone(),
        pre=torch.nn.Parameter(init.pre_init.clone()),
    )
