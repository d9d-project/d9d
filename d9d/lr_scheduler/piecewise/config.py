from typing import Annotated, Literal

from pydantic import BaseModel, Field, PositiveInt
from torch.optim import Optimizer

from d9d.core.protocol import LRSchedulerProtocol

from .builder import piecewise_schedule
from .curves import CurveBase, CurveCosine, CurveExponential, CurveLinear, CurvePoly


class CurveLinearConfig(BaseModel):
    """
    Configuration for linear interpolation.
    """

    type: Literal["linear"] = "linear"


class CurveCosineConfig(BaseModel):
    """
    Configuration for cosine interpolation.
    """

    type: Literal["cosine"] = "cosine"


class CurveExponentialConfig(BaseModel):
    """
    Configuration for exponential interpolation.
    """

    type: Literal["exponential"] = "exponential"


class CurvePolyConfig(BaseModel):
    """
    Configuration for polynomial interpolation.

    Attributes:
        power: The exponent of the polynomial function.
    """

    type: Literal["poly"] = "poly"
    power: float = 2.0


AnyCurveConfig = Annotated[
    CurveLinearConfig | CurveCosineConfig | CurveExponentialConfig | CurvePolyConfig, Field(discriminator="type")
]


def curve_from_config(config: AnyCurveConfig) -> CurveBase:
    """
    Instantiates a concrete curve object from its configuration.

    Args:
        config: The configuration object.

    Returns:
        The instantiated curve.
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
    """
    Configuration for a phase defined by a fixed number of steps.

    Attributes:
        mode: Discriminator field, must be "steps".
        steps: The absolute duration of this phase in steps.
        target_multiplier: The multiplier value at the end of this phase.
        curve: The interpolation curve configuration.
    """

    mode: Literal["steps"] = "steps"

    steps: PositiveInt
    target_multiplier: float
    curve: AnyCurveConfig


class PercentagePhaseConfig(BaseModel):
    """
    Configuration for a phase that lasts until a specific percentage of training is complete.

    Attributes:
        mode: Discriminator field, must be "percentage".
        percentage: The target progress (0.0 to 1.0) where this phase ends.
        target_multiplier: The multiplier value at the end of this phase.
        curve: The interpolation curve configuration.
    """

    mode: Literal["percentage"] = "percentage"

    percentage: float = Field(..., ge=0.0, le=1.0)
    target_multiplier: float
    curve: AnyCurveConfig


class RestPhaseConfig(BaseModel):
    """
    Configuration for a phase that fills the remainder of the training duration.

    Attributes:
        mode: Discriminator field, must be "rest".
        target_multiplier: The multiplier value at the very end of training.
        curve: The interpolation curve configuration.
    """

    mode: Literal["rest"] = "rest"

    target_multiplier: float
    curve: AnyCurveConfig


PhaseConfig = Annotated[StepPhaseConfig | PercentagePhaseConfig | RestPhaseConfig, Field(discriminator="mode")]


class PiecewiseSchedulerConfig(BaseModel):
    """
    Declarative configuration for a piecewise learning rate scheduler.

    Attributes:
        initial_multiplier: The starting learning rate multiplier.
        phases: A sequential list of phase configurations.
    """

    initial_multiplier: float
    phases: list[PhaseConfig]


def piecewise_scheduler_from_config(
    config: PiecewiseSchedulerConfig, optimizer: Optimizer, total_steps: int | None
) -> LRSchedulerProtocol:
    """
    Constructs a PyTorch scheduler from the provided configuration.

    Args:
        config: The scheduler configuration.
        optimizer: The optimizer to wrap.
        total_steps: The total number of training steps. Required if using percentage-based phases.

    Returns:
        A configured learning rate scheduler.
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
