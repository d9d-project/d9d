import pytest
import torch
from d9d.core.dist_context import DeviceMeshParameters
from d9d.loop.component.gradient_manager import GradientManager
from d9d.loop.component.model_stage_factory import TrackedModules
from d9d.loop.config import GradientManagerConfig
from torch import nn


@pytest.mark.local
def test_install_cleans_up_after_error(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    module = nn.Linear(4, 4)
    tracked = TrackedModules(dist_ctx, [module], lambda _name, _tensor: True)
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
