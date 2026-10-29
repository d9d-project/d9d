import pytest
import torch
from d9d.model_state.mapper import StateGroup
from d9d.module.block.moe import GroupedLinear
from d9d.peft import inject_peft_and_freeze, merge_peft
from d9d.peft.all import peft_method_from_config
from d9d.peft.lora import LoRAGroupedLinear, LoRALinear
from torch import nn
from torch.optim import SGD


@pytest.mark.local
def test_structural(model_for_peft, peft_config_stack):
    peft_method = peft_method_from_config(peft_config_stack)
    mapper = inject_peft_and_freeze(peft_method, model_for_peft)

    # Test structure
    for layer in range(2):
        assert isinstance(model_for_peft.layers[layer].proj, LoRALinear)
        assert isinstance(model_for_peft.layers[layer].untouched, nn.Linear)
        assert isinstance(model_for_peft.layers[layer].experts, LoRAGroupedLinear)
    assert isinstance(model_for_peft.head, nn.Linear)

    # Test grads
    for layer in range(2):
        assert model_for_peft.layers[layer].proj.lora_A.weight.requires_grad
        assert model_for_peft.layers[layer].proj.lora_B.weight.requires_grad
        assert not model_for_peft.layers[layer].proj.base.weight.requires_grad
        assert not model_for_peft.layers[layer].untouched.weight.requires_grad
    assert model_for_peft.head.weight.requires_grad

    # Test mapper
    assert mapper.state_dependency_groups() == frozenset(
        {
            StateGroup(inputs=frozenset({"layers.1.proj.weight"}), outputs=frozenset({"layers.1.proj.base.weight"})),
            StateGroup(
                inputs=frozenset({"layers.1.experts.weight"}), outputs=frozenset({"layers.1.experts.base.weight"})
            ),
            StateGroup(
                inputs=frozenset({"layers.0.experts.weight"}), outputs=frozenset({"layers.0.experts.base.weight"})
            ),
            StateGroup(inputs=frozenset({"layers.0.proj.weight"}), outputs=frozenset({"layers.0.proj.base.weight"})),
        }
    )

    # Reverse test structure
    merge_peft(peft_method, model_for_peft)
    for layer in range(2):
        assert isinstance(model_for_peft.layers[layer].proj, nn.Linear)
        assert isinstance(model_for_peft.layers[layer].untouched, nn.Linear)
        assert isinstance(model_for_peft.layers[layer].experts, GroupedLinear)
    assert isinstance(model_for_peft.head, nn.Linear)


@pytest.mark.local
def test_e2e(peft_config_stack, model_for_peft):
    model = model_for_peft
    x = torch.ones(4, 32).cuda().bfloat16()

    orig_parameter_names_and_shape = {(name, tuple(val.shape)) for name, val in model.named_parameters()}
    with torch.no_grad():
        orig_loss = model(x)

    peft = peft_method_from_config(peft_config_stack)
    inject_peft_and_freeze(peft, model)

    peft_trainable_parameter_names_and_shape = {
        (name, tuple(val.shape)) for name, val in model.named_parameters() if val.requires_grad
    }
    peft_trainable_parameter_names_and_shape_should_be = {
        ("head.weight", (10, 32)),
        ("layers.0.experts.lora_A.weight", (4, 64, 4)),
        ("layers.0.experts.lora_B.weight", (4, 4, 32)),
        ("layers.0.proj.lora_A.weight", (4, 32)),
        ("layers.0.proj.lora_B.weight", (64, 4)),
        ("layers.1.experts.lora_A.weight", (4, 64, 4)),
        ("layers.1.experts.lora_B.weight", (4, 4, 32)),
        ("layers.1.proj.lora_A.weight", (4, 32)),
        ("layers.1.proj.lora_B.weight", (64, 4)),
    }
    assert peft_trainable_parameter_names_and_shape == peft_trainable_parameter_names_and_shape_should_be

    with torch.no_grad():
        loss_lora_untrained = model(x)

    assert torch.allclose(loss_lora_untrained, orig_loss, atol=1e-3, rtol=0)

    for _ in range(100):
        opt = SGD(model.parameters(), lr=1e-2)
        loss_tmp = model(x)
        loss_tmp.backward()
        opt.step()
        opt.zero_grad()

    with torch.no_grad():
        loss_lora_trained = model(x)

    assert (loss_lora_untrained - loss_lora_trained).item() >= 0.5

    merge_peft(peft, model)
    with torch.no_grad():
        loss_merged_trained = model(x)
    assert torch.allclose(loss_merged_trained, loss_lora_trained, atol=1e-3, rtol=0)
    assert {(name, tuple(val.shape)) for name, val in model.named_parameters()} == orig_parameter_names_and_shape
