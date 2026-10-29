import pytest
import torch
from d9d.core.autograd import GLOBAL_GRAD_CONTEXT, GradDirection
from d9d.module.block.moe import GroupedLinear
from d9d.module.block.moe.grouped_mm import ExpertTokenGroups


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
        n_groups * batch_size_per_group, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    counts = torch.tensor([batch_size_per_group] * n_groups, dtype=torch.long)
    x_groups = ExpertTokenGroups(counts.cuda(), host_counts=counts)

    out = model(x, x_groups)
    loss = out.sum()

    if direction is not None:
        directions = [direction]
    else:
        # Ensure both are enabled for the 'None' case test
        directions = [GradDirection.inputs, GradDirection.weight]

    with GLOBAL_GRAD_CONTEXT.with_directions(*directions):
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


@pytest.mark.local
def test_with_features_keeps_backend_device_and_dtype():
    from d9d.module.block.moe.grouped_mm import EagerGroupedMmBackendConfig
    from d9d.module.block.moe.grouped_mm.impl.eager import EagerGroupedMm

    # an explicit backend that auto-detection would not pick on this machine
    base = GroupedLinear(
        4, 64, 32, grouped_mm_backend=EagerGroupedMmBackendConfig(), device="cuda", dtype=torch.bfloat16
    )

    derived = base.with_features(64, 8)

    assert isinstance(derived.kernel, EagerGroupedMm)
    assert derived.n_groups == 4
    assert (derived.in_features, derived.out_features) == (64, 8)
    assert derived.weight.shape == (4, 64, 8)
    assert derived.weight.device == base.weight.device
    assert derived.weight.dtype == torch.bfloat16
