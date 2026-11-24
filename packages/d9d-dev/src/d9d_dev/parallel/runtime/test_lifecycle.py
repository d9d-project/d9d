import pytest
import torch
from d9d.dist_context import DENSE_DOMAIN, DenseDim, DenseDomainConfig, DeviceMeshParameters
from d9d.pipelining.api import mark_parameter_group, marked_parameter_groups
from torch import nn
from torch.distributed.tensor import Shard

from d9d_dev.parallel.runtime.helpers import distribute_parameters, make_runtime, run_one_region


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
def test_wait_finishes_each_step_once():
    runtime = make_runtime()
    parameters = _parameters()
    calls = []

    def double_gradient(parameter):
        calls.append("double")
        parameter.grad.mul_(2)

    parameters["used"].register_post_accumulate_grad_hook(double_gradient)
    runtime.register(parameters, [])

    with runtime.bind() as bound:
        for step, use_optional in enumerate((False, True)):
            bound.zero_grad()
            run_one_region(bound, parameters, lambda use_optional=use_optional: _backward(parameters, use_optional))
            with pytest.raises(ValueError, match=r"already finished.*Call zero_grad\(\)"):
                bound.wait()

            assert len(calls) == step + 1
            torch.testing.assert_close(parameters["used"].grad, torch.full_like(parameters["used"], 2))
            # An unused parameter gets a zero gradient.
            expected = (
                torch.ones_like(parameters["optional"]) if use_optional else torch.zeros_like(parameters["optional"])
            )
            torch.testing.assert_close(parameters["optional"].grad, expected)


@pytest.mark.local
def test_failed_hook_finishes_the_step():
    runtime = make_runtime()
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
    runtime.register(parameters, [])

    with runtime.bind() as bound:
        bound.zero_grad()
        with pytest.raises(RuntimeError, match="Injected hook failure"):
            run_one_region(bound, parameters, lambda: _backward(parameters, use_optional=False))
        # A retry must not run the first hook again.
        with pytest.raises(ValueError, match="already finished"):
            bound.wait()
        assert calls == ["first", "failed"]

        failing_hook.remove()
        bound.zero_grad()
        run_one_region(bound, parameters, lambda: _backward(parameters, use_optional=False))
        assert calls == ["first", "failed", "first"]


@pytest.mark.local
def test_wait_preconditions_can_be_corrected():
    runtime = make_runtime()
    parameters = _parameters()
    runtime.register(parameters, [])

    with runtime.bind() as bound:
        bound.zero_grad()
        module_parameters = bound.parameters(parameters)
        with module_parameters.group_weights(parameters, None):
            _backward(parameters, use_optional=False)
        with pytest.raises(ValueError, match="did not reduce the parameter groups"):
            bound.wait()

        bound.zero_grad()
        module_parameters.record_order((parameters,))
        with module_parameters.backward(final=True).region(0):
            _backward(parameters, use_optional=False)
        expected = parameters["used"].grad
        parameters["used"].grad = torch.empty_like(expected)
        with pytest.raises(ValueError, match="was replaced"):
            bound.wait()

        parameters["used"].grad = expected
        bound.wait()
        torch.testing.assert_close(parameters["used"].grad, torch.ones_like(parameters["used"]))


class _Groups(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = mark_parameter_group(nn.Linear(4, 4, device="cuda"))
        self.second = mark_parameter_group(nn.Linear(4, 4, device="cuda"))

    async def forward_steps(self, steps, inputs):
        hidden = await steps.call(self.first, inputs)
        return await steps.call(self.second, hidden)


def _record_twice(runtime, bound, model):
    bound.parameters(model).record_order((model, model.first, model.second))
    bound.parameters(model).record_order((model, model.second, model.first))


def _use_parameters_after_rebind(runtime, bound, model):
    parameters = bound.parameters(model)
    bound.close()
    with runtime.bind(), parameters.group_weights(model, None):
        pass


def _register_foreign_group(runtime, bound, model):
    bound.close()
    runtime.register(nn.Linear(4, 4), [model.first])


def _tie_across_modules(runtime, bound, model):
    other = nn.Linear(4, 4, device="cuda")
    other.weight = model.first.weight
    bound.close()
    runtime.register(other, [])
    runtime.bind()


def _step_after_close(runtime, bound, model):
    bound.close()
    bound.zero_grad()


@pytest.mark.local
@pytest.mark.parametrize(
    ("call", "match"),
    [
        pytest.param(
            lambda runtime, bound, model: bound.parameters(nn.Linear(4, 4)),
            r"module \(Linear\) is not registered",
            id="unknown_module",
        ),
        pytest.param(
            lambda runtime, bound, model: bound.parameters(model).record_order((nn.Linear(4, 4),)),
            r"module \(Linear\) is not the root of a parameter group",
            id="unknown_group",
        ),
        pytest.param(
            lambda runtime, bound, model: bound.parameters(model).record_order((model, model.first)),
            r"omitted the parameters",
            id="omitted",
        ),
        pytest.param(
            _record_twice,
            r"order \(\['_Groups', 'second', 'first'\]\) of the module \(_Groups\) differs",
            id="changed_order",
        ),
        pytest.param(
            lambda runtime, bound, model: bound.parameters(model).backward(final=True),
            r"no recorded order",
            id="backward",
        ),
        pytest.param(
            lambda runtime, bound, model: runtime.register(nn.Linear(4, 4), []),
            r"Close the BoundParameters first",
            id="register_while_bound",
        ),
        pytest.param(
            lambda runtime, bound, model: runtime.bind(),
            r"already bound. Close the BoundParameters first",
            id="bind_twice",
        ),
        pytest.param(_tie_across_modules, r"Place tied parameters in one registered module", id="tied_across_modules"),
        pytest.param(
            lambda runtime, bound, model: bound.scale_grads(torch.ones((), device="cuda")),
            r"Call wait\(\) before scale_grads\(\)",
            id="scale_before_wait",
        ),
        pytest.param(_use_parameters_after_rebind, r"belong to a closed bind", id="parameters_after_rebind"),
        pytest.param(_step_after_close, r"bound parameters are closed. Call bind\(\) again", id="step_after_close"),
        pytest.param(_register_foreign_group, r"is not a submodule of the module", id="foreign_group"),
    ],
)
def test_runtime_rejects_misuse(call, match):
    runtime = make_runtime()
    model = _Groups()
    runtime.register(model, marked_parameter_groups(model))
    with runtime.bind() as bound, pytest.raises(ValueError, match=match):
        call(runtime, bound, model)


@pytest.mark.local
def test_registration_and_close_are_checked():
    runtime = make_runtime()
    module = nn.Linear(4, 4, device="cuda")
    runtime.register(module, [])
    with pytest.raises(ValueError, match="already registered"):
        runtime.register(module, [])
    bound = runtime.bind()
    bound.close()
    # A second close does nothing.
    bound.close()
    for call in (lambda: bound.parameters(module), lambda: bound.scale_grads(torch.ones(())), bound.wait):
        with pytest.raises(ValueError, match=r"Call bind\(\) again"):
            call()


@pytest.mark.distributed
def test_bind_rejects_shards_that_it_cannot_gather(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8, dense=DenseDomainConfig(data_parallel_shard=8)))
    model = nn.Linear(16, 16, bias=False, device="cuda")
    distribute_parameters(model, dist_ctx.mesh_for(DENSE_DOMAIN)[DenseDim.DP_CP_SHARD], (Shard(1),))
    runtime = make_runtime()
    runtime.register(model, [])
    with pytest.raises(ValueError, match="cannot gather"):
        runtime.bind()
