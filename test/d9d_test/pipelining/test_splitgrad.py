import gc

import pytest
import torch
from d9d.pipelining.infra.stage.splitgrad import (
    stage_backward_full,
    stage_backward_input,
    stage_backward_weight,
)

from d9d_test.pipelining.definitions import (
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

    loss = model(x, y)["x"].mean()

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

    loss = model(x, y)["x"].mean()

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
