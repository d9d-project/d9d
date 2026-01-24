import pytest
import torch
from d9d.peft.lora import LoRALinear, LoRAParameters
from torch import nn


@pytest.mark.local
def test_linear_math():
    """Verify the arithmetic within a LoRALinear layer."""
    base = nn.Linear(10, 20, bias=False)
    # Ensure known weights
    nn.init.constant_(base.weight, 1.0)

    params = LoRAParameters(r=2, alpha=4, dropout=0.0)
    lora_layer = LoRALinear(base, params)

    # Initialize LORA weights determinstically
    nn.init.constant_(lora_layer.lora_A.weight, 0.5)
    nn.init.constant_(lora_layer.lora_B.weight, 0.1)

    # Input
    x = torch.ones(1, 10)

    # Forward
    # lora_A(x) = x @ A.T. Shape (1, 10) @ (10, 2) = (1, 2). Values: 1*0.5*10 = 5.0
    # lora_B(A(x)) = (1, 2) @ B.T (2, 20). Values: 5.0 * 0.1 * 2 = 1.0
    # scale = alpha/r = 4/2 = 2.0
    # lora_term = 1.0 * 2.0 = 2.0
    # base(x) = x @ W.T (10, 20). Values: 1.0 * 1.0 * 10 = 10.0
    # result = 10.0 + 2.0 = 12.0

    y = lora_layer(x)
    assert torch.allclose(y, torch.full_like(y, 12.0))

    # Merge
    merged_base = lora_layer.merge_with_base_()
    assert merged_base is base

    # New base weight should correspond to effective weight
    # W_new = W + scale * (B @ A)
    # B (20, 2), A (2, 10) -> B@A is (20, 10) matrix of (0.1*0.5*2 = 0.1)
    # W_new = 1.0 + 2.0 * 0.1 = 1.2

    y_merged = base(x)
    # expected: 1.2 * 10 = 12.0
    assert torch.allclose(y_merged, torch.full_like(y_merged, 12.0))
    assert torch.allclose(base.weight, torch.full_like(base.weight, 1.2))
