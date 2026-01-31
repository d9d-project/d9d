import pytest
import torch
from d9d.lr_scheduler.piecewise import CurveLinear, piecewise_schedule
from torch.nn import Linear
from torch.optim import SGD


@pytest.fixture
def optimizer():
    return SGD(Linear(1, 1).parameters(), lr=0.1)


@pytest.mark.local
def test_success(optimizer):
    scheduler = (
        piecewise_schedule(initial_multiplier=0.0, total_steps=10)
        .for_steps(2, 1.0, CurveLinear())
        .until_percentage(0.5, 0.1, CurveLinear())
        .fill_rest(0.01, CurveLinear())
        .build(optimizer)
    )

    lrs = []
    # LambdaLR sets the initial LR immediately upon init
    lrs.append(optimizer.param_groups[0]["lr"])

    for _ in range(10):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    # Expected values calculation
    # Base LR = 0.1

    torch.testing.assert_close(
        torch.tensor(lrs),
        torch.tensor([0.0, 0.05, 0.1, 0.07, 0.04, 0.01, 0.0082, 0.0064, 0.0046, 0.0028, 0.001])
    )


@pytest.mark.local
def test_builder_percentage_no_total_steps():
    with pytest.raises(ValueError, match="define 'total_steps'"):
        piecewise_schedule(0.0).until_percentage(0.5, 1.0, CurveLinear())


@pytest.mark.local
def test_builder_percentage_out_of_bounds():
    builder = piecewise_schedule(0.0, total_steps=100)
    with pytest.raises(ValueError, match="range"):
        builder.until_percentage(1.5, 1.0, CurveLinear())


@pytest.mark.local
def test_builder_percentage_backward_step():
    builder = piecewise_schedule(0.0, total_steps=100)
    builder.for_steps(50, 1.0, CurveLinear())  # now at step 50

    with pytest.raises(ValueError, match="is behind current cursor"):
        builder.until_percentage(0.4, 0.5, CurveLinear())  # 0.4 * 100 = 40 < 50


@pytest.mark.local
def test_builder_exceed_total_steps(optimizer):
    builder = piecewise_schedule(0.0, total_steps=10)
    builder.for_steps(15, 1.0, CurveLinear())

    with pytest.raises(ValueError, match="defined for 15 steps, but total_steps is 10"):
        builder.build(optimizer)
