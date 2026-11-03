import gc

import pytest
import torch
from d9d.autograd.split_backward import (
    stage_backward_full,
    stage_backward_input,
    stage_backward_weight,
)

from d9d_dev.pipelining.definitions import (
    _Shared,
    _Transfer,
    build_pp_inputs,
    build_pp_model,
    check_pp_hooks_ran,
    do_standard_backward,
    register_pp_hooks,
)


@pytest.mark.local
def test_custom_backward_correctness():
    model = build_pp_model()

    x, y = build_pp_inputs(x_with_grad=True)

    orig_snapshot = do_standard_backward(model, x, y)

    hook_state = register_pp_hooks(model)

    loss = model(_Transfer(x=x), _Shared(y=y)).x.mean()

    results = stage_backward_full(outputs=[loss], output_grads=[torch.ones_like(loss)], inputs=[x, y])

    # check that we do not set input .grad variables - these are not needed to be stored at regular .grad container
    assert x.grad is None
    assert y.grad is None

    check_pp_hooks_ran(hook_state, 1)

    assert len(results) == 2
    assert torch.allclose(results[0], orig_snapshot["x"])
    assert results[1] is None

    assert torch.allclose(model.w1.grad, orig_snapshot["w1"])
    assert torch.allclose(model.w2.grad, orig_snapshot["w2"])
    assert torch.allclose(model.w3.grad, orig_snapshot["w3"])


@pytest.mark.local
def test_split_backward_correctness():
    model = build_pp_model()

    x, y = build_pp_inputs(x_with_grad=True)

    orig_snapshot = do_standard_backward(model, x, y)

    loss = model(_Transfer(x=x), _Shared(y=y)).x.mean()

    hook_state = register_pp_hooks(model)

    input_grads, pending = stage_backward_input(
        outputs=[loss], output_grads=[torch.ones_like(loss)], inputs=[x, y], weights=model.parameters()
    )

    check_pp_hooks_ran(hook_state, 0)

    # check that we do not set input .grad variables - these are not needed to be stored at regular .grad container
    assert x.grad is None
    assert y.grad is None

    # Check input gradients immediately
    assert torch.allclose(input_grads[0], orig_snapshot["x"]), "Input gradients mismatch in split phase"
    assert input_grads[1] is None

    # Check weights are NOT updated yet
    assert model.w1.grad is None
    assert model.w2.grad is None
    assert model.w3.grad is None

    # trigger GC to simulate cleaning up unused objects
    input_grads = None
    gc.collect()

    # B. Backward Weight Phase
    stage_backward_weight(pending)

    check_pp_hooks_ran(hook_state, 1)

    # check that we still do not set input .grad variables - these are not needed to be stored at
    # regular .grad container
    assert x.grad is None
    assert y.grad is None

    assert torch.allclose(model.w1.grad, orig_snapshot["w1"])
    assert torch.allclose(model.w2.grad, orig_snapshot["w2"])
    assert torch.allclose(model.w3.grad, orig_snapshot["w3"])

    for group in pending.param_groups:
        # Check cleanup happens inside `stage_backward_weight` (it sets grads/intermediates to None)
        assert group.grads is None
        assert group.intermediates is None


class _AuxThenProduct(torch.autograd.Function):
    """Returns ``(aux, x @ w)``: the differentiable output is not the first one."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        y = x @ w
        aux = y.detach().clone()
        ctx.mark_non_differentiable(aux)
        ctx.set_materialize_grads(False)
        return aux, y

    @staticmethod
    def backward(ctx, _, grad_y):
        x, w = ctx.saved_tensors
        return grad_y @ w.T, x.T @ grad_y


class _ProductAndDouble(torch.autograd.Function):
    """Returns ``(x @ w, 2 * x @ w)``: two differentiable outputs of the same shape."""

    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        y = x @ w
        return y, 2 * y

    @staticmethod
    def backward(ctx, grad_y, grad_double):
        x, w = ctx.saved_tensors
        grad = grad_y + 2 * grad_double
        return grad @ w.T, x.T @ grad


def _aux_then_product_loss(x, w):
    _, y = _AuxThenProduct.apply(x, w)
    return (y * y).sum()


def _product_and_double_loss(x, w):
    y, double = _ProductAndDouble.apply(x, w)
    # different upstream gradients for the two outputs, so mixing them up changes the result
    return (y * y).sum() + double.sum()


@pytest.mark.local
@pytest.mark.parametrize("loss_fn", [_aux_then_product_loss, _product_and_double_loss])
def test_split_backward_multi_output_node(loss_fn):
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 4))
    x = torch.randn(16, 8, requires_grad=True)

    loss_fn(x, w).backward()
    expected_x_grad, expected_w_grad = x.grad, w.grad
    x.grad, w.grad = None, None

    loss = loss_fn(x, w)
    input_grads, pending = stage_backward_input(
        outputs=[loss], output_grads=[torch.ones_like(loss)], inputs=[x], weights=[w]
    )
    stage_backward_weight(pending)

    assert torch.allclose(input_grads[0], expected_x_grad)
    assert torch.allclose(w.grad, expected_w_grad)


def _reused_twice_loss(x, w):
    return ((x @ w) @ w).square().sum()


def _reused_thrice_loss(x, w):
    return (((x @ w).tanh() @ w).tanh() @ w).sum()


@pytest.mark.local
@pytest.mark.parametrize(("loss_fn", "num_uses"), [(_reused_twice_loss, 2), (_reused_thrice_loss, 3)])
def test_split_backward_weight_reused_downstream(loss_fn, num_uses):
    # each use of ``w`` is an intermediate of the same group, and every later one lies downstream of the earlier ones
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(8, 8))
    x = torch.randn(16, 8, requires_grad=True)

    loss_fn(x, w).backward()
    expected_x_grad, expected_w_grad = x.grad, w.grad
    x.grad, w.grad = None, None

    loss = loss_fn(x, w)
    input_grads, pending = stage_backward_input(
        outputs=[loss], output_grads=[torch.ones_like(loss)], inputs=[x], weights=[w]
    )
    assert [len(group.intermediates) for group in pending.param_groups] == [num_uses]
    stage_backward_weight(pending)

    assert torch.allclose(input_grads[0], expected_x_grad)
    assert torch.allclose(w.grad, expected_w_grad)


@pytest.mark.local
def test_split_backward_weight_reused_around_another_weight():
    # ``wa`` and ``wb`` form separate groups, but the path from the second use of ``wa`` down to the first one
    # crosses the intermediate of ``wb``
    torch.manual_seed(0)
    wa = torch.nn.Parameter(torch.randn(8, 8))
    wb = torch.nn.Parameter(torch.randn(8, 8))
    x = torch.randn(16, 8, requires_grad=True)

    def loss_fn():
        return (((x @ wa).tanh() @ wb).tanh() @ wa).sum()

    loss_fn().backward()
    expected_x_grad, expected_wa_grad, expected_wb_grad = x.grad, wa.grad, wb.grad
    x.grad, wa.grad, wb.grad = None, None, None

    loss = loss_fn()
    input_grads, pending = stage_backward_input(
        outputs=[loss], output_grads=[torch.ones_like(loss)], inputs=[x], weights=[wa, wb]
    )
    assert sorted(len(group.intermediates) for group in pending.param_groups) == [1, 2]
    stage_backward_weight(pending)

    assert torch.allclose(input_grads[0], expected_x_grad)
    assert torch.allclose(wa.grad, expected_wa_grad)
    assert torch.allclose(wb.grad, expected_wb_grad)


def _independent_output(shared_weight: bool):
    # The second output does not depend on the inputs, so the input backward never reaches it.
    weight = torch.nn.Parameter(torch.randn(4, 4))
    independent = torch.nn.Parameter(torch.randn(4, 4))
    inputs = torch.randn(2, 4, requires_grad=True)

    def forward():
        transformed = weight.square()
        return [inputs @ transformed, (transformed if shared_weight else independent).square()]

    return forward, [torch.randn(2, 4), torch.randn(4, 4)], [inputs], [weight, independent]


def _overlapping_groups():
    # ``shared`` joins the groups of ``first`` and ``second``, so all three end up in one group.
    first = torch.nn.Parameter(torch.randn(4))
    second = torch.nn.Parameter(torch.randn(4))
    shared = torch.nn.Parameter(torch.randn(4))
    inputs = [torch.randn(4, requires_grad=True), torch.randn(4, requires_grad=True)]

    def forward():
        return [(inputs[0] * (first + shared) + inputs[1] * (second + shared)).sum()]

    return forward, None, inputs, [first, second, shared]


@pytest.mark.local
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(lambda: _independent_output(shared_weight=False), id="independent_output"),
        pytest.param(lambda: _independent_output(shared_weight=True), id="independent_output_shared_weight"),
        pytest.param(_overlapping_groups, id="overlapping_groups"),
    ],
)
def test_split_backward_matches_full_backward(case):
    forward, output_grads, inputs, weights = case()
    torch.autograd.backward(forward(), output_grads)
    expected_inputs = [value.grad for value in inputs]
    expected_weights = [weight.grad for weight in weights]
    for value in (*inputs, *weights):
        value.grad = None

    input_grads, pending = stage_backward_input(forward(), output_grads, inputs, weights)
    assert all(weight.grad is None for weight in weights)
    stage_backward_weight(pending)
    with pytest.raises(ValueError, match="already ran"):
        stage_backward_weight(pending)

    for actual, expected in zip(input_grads, expected_inputs, strict=True):
        torch.testing.assert_close(actual, expected)
    for weight, expected in zip(weights, expected_weights, strict=True):
        torch.testing.assert_close(weight.grad, expected)
