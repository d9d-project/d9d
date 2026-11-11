import pytest
import torch
from d9d.dist_context import BASE_DOMAIN, BaseDim, DeviceMeshParameters
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
    tracked = TrackedModules(dist_ctx, {0: module}, lambda _name, _tensor: True, parameter_runtime=runtime)
    manager = GradientManager(dist_ctx, tracked, GradientManagerConfig(), parameter_runtime=runtime)

    with pytest.raises(RuntimeError, match="step failed"), manager.install():
        module(torch.randn(2, 4)).sum().backward()
        raise RuntimeError("step failed")

    # The failed step unbinds the runtime and clears the gradients.
    assert all(param.grad is None for param in module.parameters())

    # The next install binds again instead of failing on a bound runtime.
    with manager.install():
        pass


@pytest.mark.distributed
def test_tensor_parallel_ranks_count_once(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=4, tensor_parallel=2))
    dp_rank = dist_ctx.coordinate(BASE_DOMAIN, BaseDim.DP)
    module = nn.Linear(4, 1, bias=False, device="cuda")
    runtime = build_parameter_runtime(ParameterRuntimeConfig())
    runtime.register_stage(0, module)
    tracked = TrackedModules(dist_ctx, {0: module}, lambda _name, _tensor: True, parameter_runtime=runtime)
    manager = GradientManager(dist_ctx, tracked, GradientManagerConfig(), parameter_runtime=runtime)

    # The ranks of one tensor-parallel group take the same samples.
    inputs = torch.full((3, 4), float(dp_rank + 1), device="cuda")
    weight = torch.tensor(3.0, device="cuda")
    with manager.install():
        plan = runtime.reduction_plan(0, (0,), final=True)
        with runtime.group_weights(0, 0, prepare=plan.before_region.get(0, ())):
            loss = module(inputs).sum()
            loss.backward()
        for bucket_ids in plan.after_region.values():
            for bucket_id in bucket_ids:
                runtime.reduce(plan, bucket_id)
        manager.add_loss_with_weight(loss.detach() / weight, weight)
        manager.sync_and_scale()

        # The weights sum over the 4 data-parallel ranks, not over all 8 ranks.
        torch.testing.assert_close(module.weight.grad, inputs.sum(dim=0, keepdim=True) / (4 * weight))
