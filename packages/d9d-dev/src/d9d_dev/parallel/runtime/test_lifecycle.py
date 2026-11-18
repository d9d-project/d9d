import pytest
import torch
from d9d.dist_context import DENSE_DOMAIN, DenseDim, DenseDomainConfig, DeviceMeshParameters
from d9d.pipelining.api import mark_parameter_group, marked_parameter_groups
from torch import nn
from torch.distributed.tensor import Shard

from d9d_dev.parallel.runtime.helpers import distribute_parameters, make_runtime, run_one_region


@pytest.fixture
def runtime():
    runtime = make_runtime()
    yield runtime
    runtime.unbind()


def _parameters():
    return nn.ParameterDict(
        {"used": nn.Parameter(torch.ones(1, device="cuda")), "optional": nn.Parameter(torch.ones(1, device="cuda"))}
    )


def _backward(parameters, use_optional):
    loss = parameters["used"].sum()
    if use_optional:
        loss = loss + parameters["optional"].sum()
    loss.backward()


@pytest.mark.local
def test_wait_finishes_each_step_once(runtime):
    parameters = _parameters()
    calls = []

    def double_gradient(parameter):
        calls.append("double")
        parameter.grad.mul_(2)

    parameters["used"].register_post_accumulate_grad_hook(double_gradient)
    runtime.register_stage(0, parameters)
    runtime.bind()

    for step, use_optional in enumerate((False, True)):
        runtime.zero_grad()
        run_one_region(runtime, lambda use_optional=use_optional: _backward(parameters, use_optional))
        with pytest.raises(ValueError, match=r"already finished.*Call zero_grad\(\)"):
            runtime.wait()

        assert len(calls) == step + 1
        torch.testing.assert_close(parameters["used"].grad, torch.full_like(parameters["used"], 2))
        # As in the gradient synchronizer, an unused parameter gets a zero gradient.
        expected = torch.ones_like(parameters["optional"]) if use_optional else torch.zeros_like(parameters["optional"])
        torch.testing.assert_close(parameters["optional"].grad, expected)


@pytest.mark.local
def test_failed_hook_finishes_the_step(runtime):
    parameters = _parameters()
    calls = []

    def first(parameter):
        calls.append("first")
        parameter.grad.mul_(2)

    def fail(_parameter):
        calls.append("failed")
        raise RuntimeError("Injected hook failure.")

    parameters["used"].register_post_accumulate_grad_hook(first)
    failing_hook = parameters["used"].register_post_accumulate_grad_hook(fail)
    runtime.register_stage(0, parameters)
    runtime.bind()

    runtime.zero_grad()
    with pytest.raises(RuntimeError, match="Injected hook failure"):
        run_one_region(runtime, lambda: _backward(parameters, use_optional=False))
    # A retry must not run the first hook again.
    with pytest.raises(ValueError, match="already finished"):
        runtime.wait()
    assert calls == ["first", "failed"]

    failing_hook.remove()
    runtime.zero_grad()
    run_one_region(runtime, lambda: _backward(parameters, use_optional=False))
    assert calls == ["first", "failed", "first"]


@pytest.mark.local
def test_wait_preconditions_can_be_corrected(runtime):
    parameters = _parameters()
    runtime.register_stage(0, parameters)
    runtime.bind()
    runtime.zero_grad()

    with runtime.stage(0).group_weights(0):
        _backward(parameters, use_optional=False)
    with pytest.raises(ValueError, match="did not reduce the parameter groups"):
        runtime.wait()

    runtime.zero_grad()
    stage = runtime.stage(0)
    stage.record_order((0,))
    with stage.backward(final=True).region(0):
        _backward(parameters, use_optional=False)
    expected = parameters["used"].grad
    parameters["used"].grad = torch.empty_like(expected)
    with pytest.raises(ValueError, match="was replaced"):
        runtime.wait()

    parameters["used"].grad = expected
    runtime.wait()
    torch.testing.assert_close(parameters["used"].grad, torch.ones_like(parameters["used"]))


class _Groups(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = mark_parameter_group(nn.Linear(4, 4, device="cuda"))
        self.second = mark_parameter_group(nn.Linear(4, 4, device="cuda"))

    async def forward_steps(self, steps, inputs):
        hidden = await steps.call(self.first, inputs)
        return await steps.call(self.second, hidden)


def _record_twice(runtime, model):
    runtime.stage(0).record_order((0, 1, 2))
    runtime.stage(0).record_order((0, 2, 1))


def _use_stage_after_rebind(runtime, model):
    stage = runtime.stage(0)
    runtime.unbind()
    runtime.bind()
    with stage.group_weights(0):
        pass


def _register_foreign_group(runtime, model):
    runtime.unbind()
    runtime.register_stage(1, nn.Linear(4, 4), [model.first])


def _tie_across_stages(runtime, model):
    other = nn.Linear(4, 4, device="cuda")
    other.weight = model.first.weight
    runtime.unbind()
    runtime.register_stage(1, other)
    runtime.bind()


@pytest.mark.local
@pytest.mark.parametrize(
    ("call", "match"),
    [
        pytest.param(lambda runtime, model: runtime.stage(1), r"stage \(1\) is not registered", id="unknown_stage"),
        pytest.param(
            lambda runtime, model: runtime.stage(0).group_index(nn.Linear(4, 4)),
            r"module \(Linear\) is not the root of a parameter group",
            id="unknown_group",
        ),
        pytest.param(
            lambda runtime, model: runtime.stage(0).record_order((0, 1)), r"omitted the parameters", id="omitted"
        ),
        pytest.param(_record_twice, r"order \(\(0, 2, 1\)\) of stage \(0\) differs", id="changed_order"),
        pytest.param(lambda runtime, model: runtime.stage(0).backward(final=True), r"no recorded order", id="backward"),
        pytest.param(
            lambda runtime, model: runtime.register_stage(1, nn.Linear(4, 4)), r"Call unbind\(\) first", id="bound"
        ),
        pytest.param(_tie_across_stages, r"Place tied parameters on one stage", id="tied_across_stages"),
        pytest.param(
            lambda runtime, model: runtime.scale_grads(torch.ones((), device="cuda")),
            r"Call wait\(\) before scale_grads\(\)",
            id="scale_before_wait",
        ),
        pytest.param(_use_stage_after_rebind, r"Call stage\(\) again after bind\(\)", id="stage_after_rebind"),
        pytest.param(_register_foreign_group, r"is not a submodule of the stage", id="foreign_group"),
    ],
)
def test_runtime_rejects_misuse(runtime, call, match):
    model = _Groups()
    runtime.register_stage(0, model, marked_parameter_groups(model))
    runtime.bind()
    with pytest.raises(ValueError, match=match):
        call(runtime, model)


@pytest.mark.local
def test_runtime_requires_bind():
    runtime = make_runtime()
    runtime.register_stage(0, nn.Linear(4, 4, device="cuda"))
    with pytest.raises(ValueError, match="already registered"):
        runtime.register_stage(0, nn.Linear(4, 4, device="cuda"))
    for call in (lambda: runtime.stage(0), lambda: runtime.scale_grads(torch.ones(())), runtime.wait):
        with pytest.raises(ValueError, match=r"Call bind\(\) first"):
            call()
    # Without a bind, there is nothing to reset or drop.
    runtime.zero_grad()
    runtime.invalidate_weights()
    runtime.unbind()


@pytest.mark.distributed
def test_bind_rejects_shards_that_it_cannot_gather(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8)))
    model = nn.Linear(16, 16, bias=False, device="cuda")
    distribute_parameters(model, dist_ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_CP_SHARD], (Shard(1),))
    runtime = make_runtime()
    runtime.register_stage(0, model)
    with pytest.raises(ValueError, match="cannot gather"):
        runtime.bind()
