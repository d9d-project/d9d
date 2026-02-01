import pytest
import torch
from d9d.core.autograd import GLOBAL_GRAD_CONTEXT, GradDirection
from d9d.module.block.moe import GroupedLinear


@pytest.fixture(autouse=True)
def reset_grad_context():
    yield
    GLOBAL_GRAD_CONTEXT.set_directions(GradDirection.inputs, GradDirection.weight)


@pytest.mark.local
@pytest.mark.parametrize(
    ("direction", "expect_input_grad", "expect_weight_grad"),
    [
        (GradDirection.inputs, True, False),
        (GradDirection.weight, False, True),
        (None, True, True),
    ],
)
def test_grouped_linear_grad_directions(direction, expect_input_grad, expect_weight_grad):
    # Setup
    n_groups = 2
    in_features = 32
    out_features = 64
    batch_size_per_group = 8

    model = GroupedLinear(n_groups, in_features, out_features, device="cuda", dtype=torch.bfloat16)

    x = torch.randn(
        n_groups * batch_size_per_group,
        in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True
    )

    x_groups = torch.tensor([batch_size_per_group] * n_groups, device="cpu", dtype=torch.long)

    out = model(x, x_groups)
    loss = out.sum()

    if direction is not None:
        GLOBAL_GRAD_CONTEXT.set_directions(direction)
    else:
        # Ensure both are enabled for the 'None' case test
        GLOBAL_GRAD_CONTEXT.set_directions(GradDirection.inputs, GradDirection.weight)

    loss.backward()

    if expect_input_grad:
        assert x.grad is not None
        assert torch.norm(x.grad) > 0
    else:
        assert x.grad is None

    if expect_weight_grad:
        assert model.weight.grad is not None
        assert torch.norm(model.weight.grad) > 0
    else:
        assert model.weight.grad is None
