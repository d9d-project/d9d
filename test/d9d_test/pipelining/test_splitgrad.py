import gc

import pytest
import torch
from d9d.pipelining.infra.stage.splitgrad import (
    stage_backward_full,
    stage_backward_input,
    stage_backward_weight,
)

from d9d_test.pipelining.definitions import (
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

    results = stage_backward_input(
        outputs=[loss], output_grads=[torch.ones_like(loss)], inputs=[x, y], weights=model.parameters()
    )

    check_pp_hooks_ran(hook_state, 0)

    # check that we do not set input .grad variables - these are not needed to be stored at regular .grad container
    assert x.grad is None
    assert y.grad is None

    # Check input gradients immediately
    assert torch.allclose(results.input_grads[0], orig_snapshot["x"]), "Input gradients mismatch in split phase"
    assert results.input_grads[1] is None

    # Check weights are NOT updated yet
    assert model.w1.grad is None
    assert model.w2.grad is None
    assert model.w3.grad is None

    # trigger GC to simulate cleaning up unused objects
    results.input_grads = None
    gc.collect()

    # B. Backward Weight Phase
    stage_backward_weight(weights=model.parameters(), param_groups=results.param_groups)

    check_pp_hooks_ran(hook_state, 1)

    # check that we still do not set input .grad variables - these are not needed to be stored at
    # regular .grad container
    assert x.grad is None
    assert y.grad is None

    assert torch.allclose(model.w1.grad, orig_snapshot["w1"])
    assert torch.allclose(model.w2.grad, orig_snapshot["w2"])
    assert torch.allclose(model.w3.grad, orig_snapshot["w3"])

    for group in results.param_groups:
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
    results = stage_backward_input(outputs=[loss], output_grads=[torch.ones_like(loss)], inputs=[x], weights=iter([w]))
    stage_backward_weight(weights=iter([w]), param_groups=results.param_groups)

    assert torch.allclose(results.input_grads[0], expected_x_grad)
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
    results = stage_backward_input(outputs=[loss], output_grads=[torch.ones_like(loss)], inputs=[x], weights=iter([w]))
    assert [len(group.intermediates) for group in results.param_groups] == [num_uses]
    stage_backward_weight(weights=iter([w]), param_groups=results.param_groups)

    assert torch.allclose(results.input_grads[0], expected_x_grad)
    assert torch.allclose(w.grad, expected_w_grad)
