import re

import pytest
import torch
from d9d.module.block.moe import GroupedLinear
from d9d.peft.all import PeftStackConfig
from d9d.peft.full_tune import FullTuneConfig
from d9d.peft.lora import LoRAConfig, LoRAParameters
from torch import nn


class SimpleBlock(nn.Module):
    def __init__(self, dim=32, num_experts=4):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 2, bias=False)
        self.untouched = nn.Linear(dim, dim * 2, bias=False)
        self.experts = GroupedLinear(
            n_groups=num_experts, in_features=dim * 2, out_features=dim, device=torch.device("cpu"), dtype=torch.float32
        )

    def forward(self, x):
        groups = torch.zeros(4, dtype=torch.long, device="cpu")
        groups[0] = x.shape[0]

        x = self.proj(x) + self.untouched(x)
        x = self.experts(x, groups)
        return x


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([SimpleBlock() for _ in range(2)])
        self.head = nn.Linear(32, 10, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x).mean()


@pytest.fixture
def model_for_peft():
    torch.manual_seed(42)
    return SimpleModel().cuda().bfloat16()


@pytest.fixture
def peft_config_stack():
    return PeftStackConfig(
        methods=[
            # 1. LoRA on standard projections
            LoRAConfig(module_name_pattern=re.compile(r".*proj.*"), params=LoRAParameters(r=4, alpha=8, dropout=0.0)),
            # 2. Full Tune on Head
            FullTuneConfig(module_name_pattern=re.compile(r".*head.*")),
            # 3. LoRA on Experts (Grouped)
            LoRAConfig(
                module_name_pattern=re.compile(r".*experts.*"), params=LoRAParameters(r=4, alpha=8, dropout=0.0)
            ),
        ]
    )
