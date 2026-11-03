import pytest
import torch
from d9d.dist_context import DeviceMeshParameters
from d9d.loop.component import TrackedModules, build_parameter_runtime
from d9d.loop.component.gradient_manager import GradientManager
from d9d.loop.config import GradientManagerConfig, ParameterRuntimeConfig
from torch import nn


@pytest.mark.local
def test_install_cleans_up_after_error(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    module = nn.Linear(4, 4)
    runtime = build_parameter_runtime(ParameterRuntimeConfig())
    runtime.register_stage(0, module)
    tracked = TrackedModules(dist_ctx, [module], lambda _name, _tensor: True, parameter_runtime=runtime)
    manager = GradientManager(dist_ctx, tracked, GradientManagerConfig(), parameter_runtime=runtime)

    with pytest.raises(RuntimeError, match="step failed"), manager.install():
        module(torch.randn(2, 4)).sum().backward()
        raise RuntimeError("step failed")

    # The failed step unbinds the runtime and clears the gradients.
    assert all(param.grad is None for param in module.parameters())

    # The next install binds again instead of failing on a bound runtime.
    with manager.install():
        pass
