from typing import Annotated, Literal

from pydantic import BaseModel, Field, PositiveInt
from torch.optim import Optimizer

from d9d.core.protocol import LRSchedulerProtocol

from .builder import piecewise_schedule
from .curves import CurveBase, CurveCosine, CurveExponential, CurveLinear, CurvePoly


class CurveLinearConfig(BaseModel):
    """Configuration for linear interpolation.

    Attributes:
        type: Discriminator field. Always ``"linear"``.
    """

    type: Literal["linear"] = "linear"


class CurveCosineConfig(BaseModel):
    """Configuration for cosine interpolation.

    Attributes:
        type: Discriminator field. Always ``"cosine"``.
    """

    type: Literal["cosine"] = "cosine"


class CurveExponentialConfig(BaseModel):
    """Configuration for exponential interpolation.

    Attributes:
        type: Discriminator field. Always ``"exponential"``.
    """

    type: Literal["exponential"] = "exponential"


class CurvePolyConfig(BaseModel):
    """Configuration for polynomial interpolation.

    Attributes:
        type: Discriminator field. Always ``"poly"``.
        power: Exponent of the polynomial.
    """

    type: Literal["poly"] = "poly"
    power: float = 2.0


AnyCurveConfig = Annotated[
    CurveLinearConfig | CurveCosineConfig | CurveExponentialConfig | CurvePolyConfig, Field(discriminator="type")
]


def curve_from_config(config: AnyCurveConfig) -> CurveBase:
    """Builds a curve from its configuration.

    Args:
        config: Curve configuration.

    Returns:
        The curve.
    """
    match config:
        case CurveLinearConfig():
            return CurveLinear()
        case CurvePolyConfig():
            return CurvePoly(config.power)
        case CurveExponentialConfig():
            return CurveExponential()
        case CurveCosineConfig():
            return CurveCosine()


class StepPhaseConfig(BaseModel):
    """Configuration for a phase defined by a fixed number of steps.

    Attributes:
        mode: Discriminator field. Always ``"steps"``.
        steps: Duration of this phase in steps.
        target_multiplier: Multiplier at the end of this phase.
        curve: Interpolation curve configuration.
    """

    mode: Literal["steps"] = "steps"

    steps: PositiveInt
    target_multiplier: float
    curve: AnyCurveConfig


class PercentagePhaseConfig(BaseModel):
    """Configuration for a phase that lasts until a given fraction of the total steps.

    Attributes:
        mode: Discriminator field. Always ``"percentage"``.
        percentage: Fraction of the total steps, from 0.0 to 1.0, at which this phase ends.
        target_multiplier: Multiplier at the end of this phase.
        curve: Interpolation curve configuration.
    """

    mode: Literal["percentage"] = "percentage"

    percentage: float = Field(..., ge=0.0, le=1.0)
    target_multiplier: float
    curve: AnyCurveConfig


class RestPhaseConfig(BaseModel):
    """Configuration for a phase that lasts until the end of training.

    Attributes:
        mode: Discriminator field. Always ``"rest"``.
        target_multiplier: Multiplier at the end of training.
        curve: Interpolation curve configuration.
    """

    mode: Literal["rest"] = "rest"

    target_multiplier: float
    curve: AnyCurveConfig


PhaseConfig = Annotated[StepPhaseConfig | PercentagePhaseConfig | RestPhaseConfig, Field(discriminator="mode")]


class PiecewiseSchedulerConfig(BaseModel):
    """Configuration for a piecewise learning rate scheduler.

    Attributes:
        initial_multiplier: Learning rate multiplier at step 0.
        phases: Phase configurations, in order.
    """

    initial_multiplier: float
    phases: list[PhaseConfig]


def piecewise_scheduler_from_config(
    config: PiecewiseSchedulerConfig, optimizer: Optimizer, total_steps: int | None
) -> LRSchedulerProtocol:
    """Builds a PyTorch learning rate scheduler from a configuration.

    Args:
        config: Scheduler configuration.
        optimizer: Optimizer whose learning rate the scheduler controls.
        total_steps: Total number of training steps. Required if ``config`` has ``"percentage"`` or
            ``"rest"`` phases.

    Returns:
        The learning rate scheduler.

    Raises:
        ValueError: If the phases do not fit ``total_steps``, as described in ``PiecewiseScheduleBuilder``.
    """
    builder = piecewise_schedule(config.initial_multiplier, total_steps)

    for phase in config.phases:
        curve = curve_from_config(phase.curve)
        match phase:
            case StepPhaseConfig():
                builder.for_steps(phase.steps, phase.target_multiplier, curve)
            case PercentagePhaseConfig():
                builder.until_percentage(phase.percentage, phase.target_multiplier, curve)
            case RestPhaseConfig():
                builder.fill_rest(phase.target_multiplier, curve)

    return builder.build(optimizer)
