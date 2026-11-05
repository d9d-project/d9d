import pytest
import torch
from torch import nn

from d9d_dev.parallel.runtime_step import make_runtime


@pytest.fixture
def runtime():
    runtime = make_runtime(memory_budget_bytes=0)
    yield runtime
    runtime.unbind()


def _parameters():
    return nn.ParameterDict(
        {"used": nn.Parameter(torch.ones(1, device="cuda")), "optional": nn.Parameter(torch.ones(1, device="cuda"))}
    )


def _backward(runtime, parameters, use_optional):
    with runtime.group_weights(0, 0, prepare=runtime.reduction_plan(0, (0,), final=True).before_region.get(0, ())):
        loss = parameters["used"].sum()
        if use_optional:
            loss = loss + parameters["optional"].sum()
        loss.backward()


def _reduce(runtime):
    plan = runtime.reduction_plan(0, (0,), final=True)
    for bucket_ids in plan.after_region.values():
        for bucket_id in bucket_ids:
            runtime.reduce(plan, bucket_id)


@pytest.mark.local
@pytest.mark.parametrize("use_optional", [False, True])
def test_wait_finishes_each_step_once(runtime, use_optional):
    parameters = _parameters()
    calls = []

    def double_gradient(parameter):
        calls.append("double")
        parameter.grad.mul_(2)

    parameters["used"].register_post_accumulate_grad_hook(double_gradient)
    runtime.register_stage(0, parameters)
    runtime.bind()

    for step, uses_optional in enumerate((use_optional, True)):
        runtime.zero_grad()
        _backward(runtime, parameters, uses_optional)
        _reduce(runtime)
        runtime.wait()
        with pytest.raises(ValueError, match=r"already finished.*Call zero_grad\(\)"):
            runtime.wait()

        assert len(calls) == step + 1
        torch.testing.assert_close(parameters["used"].grad, torch.full_like(parameters["used"], 2))
        if uses_optional:
            torch.testing.assert_close(parameters["optional"].grad, torch.ones_like(parameters["optional"]))
        else:
            assert parameters["optional"].grad is None


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
    _backward(runtime, parameters, use_optional=False)
    _reduce(runtime)
    with pytest.raises(RuntimeError, match="Injected hook failure"):
        runtime.wait()
    # A retry must not run the first hook again.
    with pytest.raises(ValueError, match="already finished"):
        runtime.wait()
    assert calls == ["first", "failed"]

    failing_hook.remove()
    runtime.zero_grad()
    _backward(runtime, parameters, use_optional=False)
    _reduce(runtime)
    runtime.wait()
    assert calls == ["first", "failed", "first"]


@pytest.mark.local
def test_wait_preconditions_can_be_corrected(runtime):
    parameters = _parameters()
    runtime.register_stage(0, parameters)
    runtime.bind()

    _backward(runtime, parameters, use_optional=False)
    with pytest.raises(ValueError, match="did not reduce"):
        runtime.wait()

    _reduce(runtime)
    expected = parameters["used"].grad
    parameters["used"].grad = torch.empty_like(expected)
    with pytest.raises(ValueError, match="was replaced"):
        runtime.wait()

    parameters["used"].grad = expected
    runtime.wait()
    torch.testing.assert_close(parameters["used"].grad, torch.ones_like(parameters["used"]))
    assert parameters["optional"].grad is None
