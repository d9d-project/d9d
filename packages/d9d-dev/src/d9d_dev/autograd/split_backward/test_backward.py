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
    results = stage_backward_input(
        outputs=[loss], output_grads=[torch.ones_like(loss)], inputs=[x], weights=iter([wa, wb])
    )
    assert sorted(len(group.intermediates) for group in results.param_groups) == [1, 2]
    stage_backward_weight(weights=iter([wa, wb]), param_groups=results.param_groups)

    assert torch.allclose(results.input_grads[0], expected_x_grad)
    assert torch.allclose(wa.grad, expected_wa_grad)
    assert torch.allclose(wb.grad, expected_wb_grad)


@pytest.mark.local
@pytest.mark.parametrize("shared_weight", [False, True])
def test_split_backward_keeps_output_branches_without_input_dependencies(shared_weight):
    torch.manual_seed(0)
    weight = torch.nn.Parameter(torch.randn(4, 4))
    independent = torch.nn.Parameter(torch.randn(4, 4))
    inputs = torch.randn(2, 4, requires_grad=True)

    def outputs():
        transformed = weight.square()
        return [inputs @ transformed, (transformed if shared_weight else independent).square()]

    gradients = [torch.randn(2, 4), torch.randn(4, 4)]
    torch.autograd.backward(outputs(), gradients)
    expected_input, expected_weight, expected_independent = inputs.grad, weight.grad, independent.grad
    inputs.grad, weight.grad, independent.grad = None, None, None

    result = stage_backward_input(outputs(), gradients, [inputs], iter([weight, independent]))
    torch.testing.assert_close(result.input_grads[0], expected_input)
    assert weight.grad is None
    assert independent.grad is None
    stage_backward_weight(iter([weight, independent]), result.param_groups)

    torch.testing.assert_close(weight.grad, expected_weight)
    torch.testing.assert_close(independent.grad, expected_independent)
    assert all(group.seeds is None for group in result.param_groups)


@pytest.mark.local
def test_split_backward_merges_all_overlapping_parameter_groups():
    torch.manual_seed(0)
    first = torch.nn.Parameter(torch.randn(4))
    second = torch.nn.Parameter(torch.randn(4))
    shared = torch.nn.Parameter(torch.randn(4))
    inputs = [torch.randn(4, requires_grad=True), torch.randn(4, requires_grad=True)]

    def loss():
        return (inputs[0] * (first + shared) + inputs[1] * (second + shared)).sum()

    loss().backward()
    expected = [parameter.grad for parameter in (first, second, shared)]
    expected_inputs = [value.grad for value in inputs]
    for value in (*inputs, first, second, shared):
        value.grad = None

    result = stage_backward_input([loss()], None, inputs, iter([first, second, shared]))
    assert len(result.param_groups) == 1
    assert len(result.param_groups[0].params) == 3
    stage_backward_weight(iter([first, second, shared]), result.param_groups)

    for actual, gradient in zip(result.input_grads, expected_inputs, strict=True):
        torch.testing.assert_close(actual, gradient)
    for parameter, gradient in zip((first, second, shared), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)
