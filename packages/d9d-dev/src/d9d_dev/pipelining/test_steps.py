import copy
from contextlib import contextmanager, nullcontext

import pytest
import torch
from d9d import pytree
from d9d.pipelining.api import ModuleSupportsWindows, WindowBoundary, WindowSteps, run_steps
from d9d.pipelining.api.groups import group_modules, window_groups
from d9d.pipelining.stage.windows import WindowChain, backward_chains, forward_chains
from torch import nn

from d9d_dev.nn.helper.tolerances import forward_tolerance_for
from d9d_dev.pipelining.window_models import _MLP, _Calls, _LocalSkip, _Manual, _Revisited

pytestmark = pytest.mark.local


def _assert_close(actual, expected):
    tolerance = forward_tolerance_for(expected.dtype)
    torch.testing.assert_close(actual, expected, rtol=tolerance.rtol, atol=tolerance.atol)


@pytest.mark.parametrize("model_type", [_Calls, _Manual, _Revisited, _LocalSkip])
@pytest.mark.parametrize("grad_enabled", [True, False])
def test_run_steps_matches_eager_with_repeated_and_tied_parameters(model_type, grad_enabled):
    model = model_type().double()
    reference = copy.deepcopy(model)
    assert isinstance(model, ModuleSupportsWindows)

    for _ in range(2):
        inputs = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        with torch.set_grad_enabled(grad_enabled):
            actual = model(inputs, None)
            expected = reference.reference(reference_inputs)
        _assert_close(actual, expected)
        assert actual.requires_grad == grad_enabled
        if grad_enabled:
            actual.sum().backward()
            expected.sum().backward()
            _assert_close(inputs.grad, reference_inputs.grad)

    for parameter, reference_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        if grad_enabled:
            _assert_close(parameter.grad, reference_parameter.grad)
        else:
            assert parameter.grad is None


class _Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, inputs, *, offset):
        self.inputs.append(inputs)
        return inputs * 2 + offset


@pytest.mark.parametrize("manual", [False, True])
def test_two_forwards_suspend_before_compute_and_use_resumed_state(manual):
    module = _Probe()

    async def forward(steps, inputs):
        if manual:
            inputs = await steps.enter(module, inputs)
            return await steps.finish(module(inputs, offset=3))
        return await steps.call(module, inputs, offset=3)

    inputs = [torch.tensor([1.0]), torch.tensor([5.0])]
    coroutines = [forward(WindowSteps(), value) for value in inputs]
    try:
        entries = [coroutine.send(None) for coroutine in coroutines]
        assert all(isinstance(event, WindowBoundary) and event.module is module for event in entries)
        assert module.inputs == []

        finishes = []
        for coroutine, entry in zip(coroutines, entries, strict=True):
            resumed = pytree.tree_map(
                lambda value: value + 1 if isinstance(value, torch.Tensor) else value, entry.state
            )
            finishes.append(coroutine.send(resumed))
        assert all(isinstance(event, WindowBoundary) and event.module is None for event in finishes)

        for original, seen, coroutine, finish in zip(inputs, module.inputs, coroutines, finishes, strict=True):
            assert torch.equal(seen, original + 1)
            assert torch.equal(finish.state, (original + 1) * 2 + 3)
            replacement = finish.state.detach().clone()
            with pytest.raises(StopIteration) as completed:
                coroutine.send(replacement)
            assert completed.value.value is replacement
    finally:
        for coroutine in coroutines:
            coroutine.close()


def test_manual_enter_returns_the_complete_replaced_state():
    first, second = nn.Identity(), nn.Identity()
    inputs = torch.tensor([1.0], requires_grad=True)

    async def forward(steps):
        hidden = await steps.enter(first, inputs)
        hidden, skip = await steps.enter(second, (hidden, hidden))
        return hidden + skip

    coroutine = forward(WindowSteps())
    try:
        entry = coroutine.send(None)
        assert entry.module is first
        replacement = inputs.detach().clone().requires_grad_()
        next_entry = coroutine.send(replacement)
        assert next_entry.module is second
        assert next_entry.state[0] is replacement
        assert next_entry.state[1] is replacement
        left, right = torch.tensor([3.0]), torch.tensor([7.0])
        with pytest.raises(StopIteration) as completed:
            coroutine.send((left, right))
        assert torch.equal(completed.value.value, left + right)
    finally:
        coroutine.close()


@pytest.mark.parametrize("finish_first", [False, True])
def test_call_requires_manual_region_to_finish(finish_first):
    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Identity()

        async def forward_steps(self, steps, inputs, shared):
            inputs = await steps.enter(self.layer, inputs)
            if finish_first:
                inputs = await steps.finish(inputs)
            return await steps.call(self.layer, inputs)

    model = Mixed()
    inputs = torch.tensor([1.0])
    if finish_first:
        assert run_steps(model, inputs, None) is inputs
    else:
        with pytest.raises(ValueError, match="Call finish"):
            run_steps(model, inputs, None)


def test_finish_requires_an_open_region():
    class Invalid:
        async def forward_steps(self, steps, inputs, shared):
            return await steps.finish(inputs)

    with pytest.raises(ValueError, match="Call enter"):
        run_steps(Invalid(), None, None)


def test_run_steps_rejects_foreign_awaitables_and_closes_the_coroutine():
    closed = []

    class ForeignAwaitable:
        def __await__(self):
            yield "foreign event"

    class Invalid:
        async def forward_steps(self, steps, inputs, shared):
            try:
                await ForeignAwaitable()
            finally:
                closed.append(True)

    with pytest.raises(TypeError, match="WindowBoundary"):
        run_steps(Invalid(), None, None)
    assert closed == [True]


def test_run_steps_propagates_module_failure_and_runs_cleanup():
    closed = []
    failure = RuntimeError("Module failed.")

    class Failing(nn.Module):
        def forward(self, inputs):
            raise failure

    class Model:
        async def forward_steps(self, steps, inputs, shared):
            try:
                return await steps.call(Failing(), inputs)
            finally:
                closed.append(True)

    with pytest.raises(RuntimeError, match="Module failed") as caught:
        run_steps(Model(), None, None)
    assert caught.value is failure
    assert closed == [True]


class _Runtime:
    def __init__(self, model):
        self.groups = window_groups(model)
        self.order = None
        self.bindings = []
        self.prefetches = []

    def group_index(self, stage, module):
        return self.groups.index(module)

    def group_parameters(self, stage, group):
        if group is None:
            return []
        modules = group_modules(self.groups[group], set(self.groups))
        return list(
            {id(p): p for module in modules for p in module.parameters(recurse=False) if p.requires_grad}.values()
        )

    @contextmanager
    def group_weights(self, stage, group, prepare=(), next_group=None):
        self.bindings.append(group)
        self.prefetches.append(next_group)
        yield

    def expected_group(self, stage, position):
        return None if self.order is None or position >= len(self.order) else self.order[position]

    def record_order(self, stage, order):
        assert self.order is None or self.order == order
        self.order = order


@pytest.mark.parametrize("model_type", [_Calls, _Manual, _Revisited, _LocalSkip])
@pytest.mark.parametrize("split", [False, True])
def test_window_backward_matches_eager_with_repeated_and_tied_parameters(model_type, split):
    model = model_type().double()
    reference = copy.deepcopy(model)
    runtime = _Runtime(model)
    inputs = [torch.randn(2, 3, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    reference_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    chains = [WindowChain(model, 0, runtime) for _ in inputs]
    for chain, value in zip(chains, inputs, strict=True):
        chain.start_forward(value, None)
    outputs = forward_chains(chains, runtime, 0, [nullcontext(), nullcontext()])
    for chain, actual, value in zip(chains, outputs, reference_inputs, strict=True):
        expected = reference.reference(value)
        _assert_close(actual, expected)
        expected.square().sum().backward()
        chain.start_backward([actual.square().sum()], None)
    if split:
        for chain in chains:
            chain.backward_input()
            chain.backward_weight(None)
    else:
        backward_chains(chains, runtime, 0, None, [nullcontext(), nullcontext()])
    for value, expected, chain in zip(inputs, reference_inputs, chains, strict=True):
        _assert_close(chain.input_grads([value])[0], expected.grad)
    for parameter, expected in zip(model.parameters(), reference.parameters(), strict=True):
        _assert_close(parameter.grad, expected.grad)
    assert runtime.order == ((1, 2, 1) if model_type is _Revisited else (1, 1, 2))


@pytest.mark.parametrize("wrong_hint", [False, True])
def test_explicit_prefetch_names_the_next_request(wrong_hint):
    class Hinted(_MLP):
        async def forward_steps(self, steps, inputs, shared):
            inputs = await steps.enter(self.first, inputs, prefetch=self.first if wrong_hint else self.last)
            hidden = self.first(inputs)
            hidden = await steps.enter(self.last, hidden)
            return self.last(hidden)

    model = Hinted()
    runtime = _Runtime(model)
    chain = WindowChain(model, 0, runtime)
    inputs = torch.randn(2, 3)
    chain.start_forward(inputs, None)
    if wrong_hint:
        with pytest.raises(ValueError, match="explicit prefetch target"):
            forward_chains([chain], runtime, 0, [nullcontext()])
    else:
        output = forward_chains([chain], runtime, 0, [nullcontext()])[0]
        _assert_close(output, model.last(model.first(inputs)))
        assert runtime.prefetches == [2, None]


def test_window_rejects_divergent_groups_before_gather_and_closes_coroutines():
    closed = []

    class Branch(_MLP):
        async def forward_steps(self, steps, inputs, shared):
            try:
                return await steps.call(self.first if shared else self.last, inputs)
            finally:
                closed.append(shared)

    model = Branch()
    runtime = _Runtime(model)
    chains = [WindowChain(model, 0, runtime) for _ in range(2)]
    for chain, shared in zip(chains, (False, True), strict=True):
        chain.start_forward(torch.randn(2, 3), shared)
    with pytest.raises(ValueError, match="same parameter groups"):
        forward_chains(chains, runtime, 0, [nullcontext(), nullcontext()])
    assert runtime.bindings == []
    assert closed == [False, True]


def test_window_rejects_differentiable_work_between_completed_calls():
    class Outside(_MLP):
        async def forward_steps(self, steps, inputs, shared):
            hidden = await steps.call(self.first, inputs)
            return await steps.call(self.last, hidden.sin())

    model = Outside()
    runtime = _Runtime(model)
    chain = WindowChain(model, 0, runtime)
    chain.start_forward(torch.randn(2, 3), None)
    with pytest.raises(ValueError, match="outside a window region"):
        forward_chains([chain], runtime, 0, [nullcontext()])


@pytest.mark.parametrize(
    "context",
    [torch.no_grad, lambda: torch.autocast("cpu", dtype=torch.bfloat16)],
    ids=["no_grad", "autocast"],
)
def test_window_rejects_a_context_across_a_boundary(context):
    class Spanning(_MLP):
        async def forward_steps(self, steps, inputs, shared):
            with context():
                hidden = await steps.call(self.first, inputs)
            return await steps.call(self.last, hidden)

    model = Spanning()
    chain = WindowChain(model, 0, _Runtime(model))
    try:
        with pytest.raises(ValueError, match="spans a boundary"):
            chain.start_forward(torch.randn(2, 3), None)
    finally:
        chain.close()
    # Closing the coroutine leaves the context, so the thread gets its modes back.
    assert torch.is_grad_enabled()
    assert not torch.is_autocast_enabled("cpu")
