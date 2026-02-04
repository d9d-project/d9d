from typing import ClassVar

import pytest
import torch
from d9d.core.autograd import GLOBAL_GRAD_CONTEXT, GradDirection
from d9d.core.autograd.grad_context import GlobalGradContext
from torch.autograd import Function


@pytest.fixture(autouse=True)
def reset_grad_context():
    yield
    # Clear matmul log
    ControlledMatmul.execution_log.clear()


@pytest.mark.local
def test_default_state():
    ctx = GlobalGradContext()
    assert ctx.check_direction(GradDirection.inputs)
    assert ctx.check_direction(GradDirection.weight)
    assert ctx.check_direction(None)


@pytest.mark.local
@pytest.mark.parametrize(
    ("enabled_dirs", "check_dir", "expected"),
    [
        # Only inputs enabled
        ([GradDirection.inputs], GradDirection.inputs, True),
        ([GradDirection.inputs], GradDirection.weight, False),
        # Only weights enabled
        ([GradDirection.weight], GradDirection.weight, True),
        ([GradDirection.weight], GradDirection.inputs, False),
        # Nothing enabled
        ([], GradDirection.inputs, False),
        ([], GradDirection.weight, False),
        # Both enabled
        ([GradDirection.inputs, GradDirection.weight], GradDirection.inputs, True),
        ([GradDirection.inputs, GradDirection.weight], GradDirection.weight, True),
        # Check None behavior
        ([GradDirection.inputs], None, True),
        ([], None, True),
    ],
)
def test_check_direction(enabled_dirs, check_dir, expected):
    ctx = GlobalGradContext()
    with ctx.with_directions(*enabled_dirs):
        assert ctx.check_direction(check_dir) is expected


@pytest.mark.local
def test_set_directions_overwrite():
    ctx = GlobalGradContext()

    # Step 1: Enable Inputs only
    with ctx.with_directions(GradDirection.inputs):
        assert ctx.check_direction(GradDirection.inputs)
        assert not ctx.check_direction(GradDirection.weight)

    # Step 2: Enable Weights only (should remove Inputs)
    with ctx.with_directions(GradDirection.weight):
        assert not ctx.check_direction(GradDirection.inputs)
        assert ctx.check_direction(GradDirection.weight)


class ControlledMatmul(Function):
    execution_log: ClassVar[list[str]] = []

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None

        if GLOBAL_GRAD_CONTEXT.check_direction(GradDirection.inputs) and ctx.needs_input_grad[0]:
            ControlledMatmul.execution_log.append("calc_inputs")
            grad_a = grad_output @ b.transpose(-1, -2)

        if GLOBAL_GRAD_CONTEXT.check_direction(GradDirection.weight) and ctx.needs_input_grad[1]:
            ControlledMatmul.execution_log.append("calc_weights")
            grad_b = a.transpose(-1, -2) @ grad_output

        return grad_a, grad_b


@pytest.mark.local
def test_integration_inputs_only():
    a = torch.randn(2, 4, requires_grad=True)
    w = torch.randn(4, 3, requires_grad=True)  # Weight

    out = ControlledMatmul.apply(a, w)
    loss = out.sum()

    # Mimic the split-backward schedule: Only want input grads now
    with GLOBAL_GRAD_CONTEXT.with_directions(GradDirection.inputs):
        loss.backward()

    # Verify log
    assert "calc_inputs" in ControlledMatmul.execution_log
    assert "calc_weights" not in ControlledMatmul.execution_log

    # Verify PyTorch received the grads (or None)
    assert a.grad is not None
    assert w.grad is None


@pytest.mark.local
def test_integration_weights_only():
    a = torch.randn(2, 4, requires_grad=True)
    w = torch.randn(4, 3, requires_grad=True)

    out = ControlledMatmul.apply(a, w)
    loss = out.sum()

    # Mimic the split-backward schedule: Only want weight grads now
    with GLOBAL_GRAD_CONTEXT.with_directions(GradDirection.weight):
        loss.backward()

    assert "calc_inputs" not in ControlledMatmul.execution_log
    assert "calc_weights" in ControlledMatmul.execution_log

    assert a.grad is None
    assert w.grad is not None


@pytest.mark.local
def test_integration_default_behavior():
    a = torch.randn(2, 4, requires_grad=True)
    w = torch.randn(4, 3, requires_grad=True)

    out = ControlledMatmul.apply(a, w)
    out.sum().backward()

    assert "calc_inputs" in ControlledMatmul.execution_log
    assert "calc_weights" in ControlledMatmul.execution_log

    assert a.grad is not None
    assert w.grad is not None
