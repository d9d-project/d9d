import copy

import pytest
import torch
from d9d.dist_context import DENSE_DOMAIN, DeviceMeshParameters
from d9d.optim.stochastic import StochasticAdamW
from torch.distributed.tensor import Replicate, Shard, distribute_tensor


@pytest.mark.local
@pytest.mark.parametrize(
    ("kwargs", "match_msg"),
    [
        ({"lr": -0.1}, r"^lr \("),
        ({"eps": -1e-6}, r"^eps \("),
        ({"betas": (-0.1, 0.999)}, r"^betas\[0\]"),
        ({"betas": (0.9, 1.0)}, r"^betas\[1\]"),
        ({"weight_decay": -0.01}, r"^weight_decay \("),
    ],
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

    with pytest.raises(ValueError, match="does not support closures"):
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
    for key in ("exp_avg", "exp_avg_sq"):
        assert optim2.state[param2][key].dtype == state_dtype
        assert torch.equal(optim1.state[param1][key], optim2.state[param2][key])

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
    # Stochastic rounding keeps the weight jumping between the bf16 neighbors of the optimum, 2**-6 apart above 2.0.
    assert abs(weight.item() - 2.0) <= 2 * 2**-6


@pytest.mark.local
def test_rounding_noise_ignores_default_generator():
    device = torch.device("cuda")
    start = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
    grad = torch.randn(64, 64, device=device, dtype=torch.float32)

    # Ranks draw different amounts from the default generator, but their replicas must round the same way.
    results = []
    for default_seed in (1, 2):
        torch.manual_seed(default_seed)
        param = start.clone().requires_grad_(True)
        param.grad_dtype = torch.float32
        param.grad = grad.clone()
        StochasticAdamW([param], lr=1e-3).step()
        results.append(param.detach())

    assert torch.equal(results[0], results[1])


@pytest.mark.local
def test_load_preserves_mixed_state_dtypes_and_input_dictionary():
    params = [torch.ones(4, dtype=torch.bfloat16) for _ in range(3)]
    groups = [
        {"params": params[:1], "state_dtype": torch.float32},
        {"params": params[1:], "state_dtype": torch.bfloat16},
    ]
    optimizer = StochasticAdamW(groups, lr=1e-3)
    for param, dtype in zip(params[:2], (torch.float32, torch.bfloat16), strict=True):
        optimizer.state[param] = {
            "step": 3,
            "exp_avg": torch.full_like(param, 0.1234567, dtype=dtype),
            "exp_avg_sq": torch.full_like(param, 0.01234567, dtype=dtype),
        }
    saved = copy.deepcopy(optimizer.state_dict())
    expected_generator = optimizer._generator.get_state().clone()
    optimizer._generator.manual_seed(999)

    optimizer.load_state_dict(saved)
    optimizer._generator.manual_seed(999)
    # Loading must leave the checkpoint reusable, including its generator state.
    optimizer.load_state_dict(saved)

    assert torch.equal(optimizer._generator.get_state(), expected_generator)
    for index, (param, dtype) in enumerate(zip(params[:2], (torch.float32, torch.bfloat16), strict=True)):
        for key in ("exp_avg", "exp_avg_sq"):
            assert optimizer.state[param][key].dtype == dtype
            assert torch.equal(optimizer.state[param][key], saved["state"][index][key])
    assert params[2] not in optimizer.state


@pytest.mark.distributed
@pytest.mark.parametrize(
    "mesh_params",
    [
        DeviceMeshParameters(data_parallel_shard=8),
        DeviceMeshParameters(data_parallel_replicate=8),
        DeviceMeshParameters(data_parallel_replicate=2, data_parallel_shard=4),
    ],
)
def test_load_preserves_fp32_distributed_moments(dist_ctx_factory, mesh_params):
    context = dist_ctx_factory(mesh_params)
    mesh = context.mesh_for(DENSE_DOMAIN)["dp_replicate", "dp_cp_shard", "cp_replicate"]
    param = distribute_tensor(
        torch.ones(16, device="cuda", dtype=torch.bfloat16), mesh, (Replicate(), Shard(0), Replicate())
    )
    optimizer = StochasticAdamW([param], lr=1e-3, state_dtype=torch.float32)
    optimizer.state[param] = {
        "step": 3,
        "exp_avg": torch.full_like(param, 0.1234567, dtype=torch.float32),
        "exp_avg_sq": torch.full_like(param, 0.01234567, dtype=torch.float32),
    }
    saved = copy.deepcopy(optimizer.state_dict())
    restored = StochasticAdamW([param], lr=1e-3, state_dtype=torch.float32)

    restored.load_state_dict(saved)

    for key in ("exp_avg", "exp_avg_sq"):
        moment = restored.state[param][key]
        assert moment.dtype == torch.float32
        assert moment.placements == param.placements
        assert torch.equal(moment.to_local(), saved["state"][0][key].to_local())
