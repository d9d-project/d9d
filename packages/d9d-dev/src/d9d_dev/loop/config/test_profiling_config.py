import pytest
from d9d.loop.config import ProfilingConfig
from pydantic import ValidationError


def _config(period_steps, warmup_steps, active_steps):
    return ProfilingConfig(
        traces_dir="traces",
        period_steps=period_steps,
        warmup_steps=warmup_steps,
        active_steps=active_steps,
    )


@pytest.mark.local
@pytest.mark.parametrize(("period_steps", "warmup_steps", "active_steps"), [(10, 5, 1), (4, 1, 3), (1, 0, 1)])
def test_accepts_a_cycle_that_fits_the_period(period_steps, warmup_steps, active_steps):
    config = _config(period_steps, warmup_steps, active_steps)

    assert config.record_shapes
    assert config.with_stack


@pytest.mark.local
@pytest.mark.parametrize(
    ("period_steps", "warmup_steps", "active_steps", "match"),
    [
        (4, 3, 2, "must cover"),
        (0, 0, 1, "greater than 0"),
        (4, -1, 1, "greater than or equal to 0"),
        (4, 1, 0, "greater than 0"),
    ],
)
def test_rejects_an_invalid_schedule(period_steps, warmup_steps, active_steps, match):
    with pytest.raises(ValidationError, match=match):
        _config(period_steps, warmup_steps, active_steps)
