import pytest
import torch
from d9d.loop.component.pipelined_optimizer import PipelinedOptimizer
from torch import nn


@pytest.mark.local
def test_init_state_creates_state_without_changing_parameters():
    param = nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([param], lr=0.5, weight_decay=0.1)
    pipelined = PipelinedOptimizer(mesh_pp=None, optimizers=[optimizer])

    pipelined.init_state()

    assert optimizer.state[param]["exp_avg"].shape == param.shape
    assert optimizer.param_groups[0]["lr"] == 0.5
    assert param.grad is None
    assert torch.equal(param.data, torch.ones(4))


@pytest.mark.local
def test_init_state_keeps_existing_state():
    param = nn.Parameter(torch.ones(4))
    param.grad = torch.ones(4)
    optimizer = torch.optim.AdamW([param], lr=0.5)
    optimizer.step()
    pipelined = PipelinedOptimizer(mesh_pp=None, optimizers=[optimizer])
    average = optimizer.state[param]["exp_avg"].clone()

    pipelined.init_state()

    assert optimizer.state[param]["step"] == 1
    assert torch.equal(optimizer.state[param]["exp_avg"], average)


@pytest.mark.local
def test_init_state_restores_rates_and_gradients_after_failure(monkeypatch):
    param = nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([param], lr=0.5)
    pipelined = PipelinedOptimizer(mesh_pp=None, optimizers=[optimizer])

    def fail():
        raise RuntimeError("Initialization failed.")

    monkeypatch.setattr(optimizer, "step", fail)

    with pytest.raises(RuntimeError, match="Initialization failed"):
        pipelined.init_state()

    assert optimizer.param_groups[0]["lr"] == 0.5
    assert param.grad is None
