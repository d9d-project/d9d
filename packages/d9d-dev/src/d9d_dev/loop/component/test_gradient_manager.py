import pytest
import torch
from d9d.dist_context import BASE_DOMAIN, BaseDim, DeviceMeshParameters
from d9d.loop.component.gradient_manager import GradientManager
from d9d.loop.component.model_stage_factory import TrackedModules
from d9d.loop.config import GradientManagerConfig
from torch import nn


@pytest.mark.local
def test_install_cleans_up_after_error(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    module = nn.Linear(4, 4)
    tracked = TrackedModules(dist_ctx, {0: module}, lambda _name, _tensor: True)
    manager = GradientManager(dist_ctx, tracked, GradientManagerConfig())

    with pytest.raises(RuntimeError, match="step failed"), manager.install():
        manager.set_required_accumulations(1)
        module(torch.randn(2, 4)).sum().backward()
        raise RuntimeError("step failed")

    # The failed step unbinds the synchronizer and clears the gradients.
    assert all(param.grad is None for param in module.parameters())

    # The next install binds again instead of failing on a bound synchronizer.
    with manager.install():
        pass


@pytest.mark.distributed
def test_tensor_parallel_ranks_count_once(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=4, tensor_parallel=2))
    dp_rank = dist_ctx.coordinate(BASE_DOMAIN, BaseDim.DP)
    module = nn.Linear(4, 1, bias=False, device="cuda")
    tracked = TrackedModules(dist_ctx, {0: module}, lambda _name, _tensor: True)
    manager = GradientManager(dist_ctx, tracked, GradientManagerConfig())

    # The ranks of one tensor-parallel group take the same samples.
    inputs = torch.full((3, 4), float(dp_rank + 1), device="cuda")
    weight = torch.tensor(3.0, device="cuda")
    with manager.install():
        manager.set_required_accumulations(1)
        loss = module(inputs).sum()
        loss.backward()
        manager.add_loss_with_weight(loss.detach() / weight, weight)
        manager.sync_and_scale()

        # The weights sum over the 4 data-parallel ranks, not over all 8 ranks.
        torch.testing.assert_close(module.weight.grad, inputs.sum(dim=0, keepdim=True) / (4 * weight))
