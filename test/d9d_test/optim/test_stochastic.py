import copy

import pytest
import torch
from d9d.optim.stochastic import StochasticAdamW


@pytest.mark.local
@pytest.mark.parametrize(
    ("kwargs", "match_msg"),
    [
        ({"lr": -0.1}, "Invalid learning rate"),
        ({"eps": -1e-6}, "Invalid epsilon value"),
        ({"betas": (-0.1, 0.999)}, "Invalid beta parameter at index 0"),
        ({"betas": (0.9, 1.0)}, "Invalid beta parameter at index 1"),
        ({"weight_decay": -0.01}, "Invalid weight_decay value"),
    ]
)
def test_init_validation(kwargs, match_msg):
    param = torch.zeros(1, dtype=torch.bfloat16)
    base_kwargs = {"params": [param], "lr": 1e-3}
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=match_msg):
        StochasticAdamW(**base_kwargs)


@pytest.mark.local
def test_step_errors():
    param = torch.zeros(10, dtype=torch.bfloat16, requires_grad=True)
    optim = StochasticAdamW([param], lr=1e-3)

    with pytest.raises(ValueError, match="Closure is not supported"):
        optim.step(closure=lambda: 1.0)

    # Test sparse gradient error
    param.grad = torch.sparse_coo_tensor([[0]], [1.0], size=(10,), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        optim.step()


@pytest.mark.local
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float32])
def test_reproducibility_and_state_dict(state_dtype, grad_dtype):
    torch.manual_seed(42)
    device = torch.device("cuda")

    param1 = torch.randn(10, 10, device=device, dtype=torch.bfloat16, requires_grad=True)
    param1.grad_dtype = grad_dtype
    optim1 = StochasticAdamW([param1], lr=1e-3, state_dtype=state_dtype)

    grad_step1 = torch.randn_like(param1, dtype=grad_dtype)
    grad_step2 = torch.randn_like(param1, dtype=grad_dtype)

    param1.grad = grad_step1.clone()
    optim1.step()

    state_dict = copy.deepcopy(optim1.state_dict())

    # Save parameters at t=1
    param_t1_snapshot = param1.clone().detach()

    param2 = param_t1_snapshot.clone().detach().to(device)
    param2.grad_dtype = grad_dtype
    param2.requires_grad = True

    optim2 = StochasticAdamW([param2], lr=1e-3, state_dtype=state_dtype)
    optim2.load_state_dict(state_dict)

    assert (optim1._generator.get_state() == optim2._generator.get_state()).all()

    param1.grad = grad_step2.clone()
    param2.grad = grad_step2.clone()

    optim1.step()
    optim2.step()

    assert torch.equal(param1, param2)


@pytest.mark.local
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_convergence_bf16(state_dtype):
    input_val = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.bfloat16)
    target = input_val * 2.0

    weight = torch.tensor([0.0], device="cuda", dtype=torch.bfloat16, requires_grad=True)

    optim = StochasticAdamW([weight], lr=0.1, state_dtype=state_dtype)

    initial_loss = torch.nn.functional.mse_loss(input_val * weight, target).item()

    for _ in range(100):
        optim.zero_grad()
        prediction = input_val * weight
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optim.step()

    final_loss = torch.nn.functional.mse_loss(input_val * weight, target).item()

    assert final_loss < initial_loss
    assert final_loss <= 1e-3
