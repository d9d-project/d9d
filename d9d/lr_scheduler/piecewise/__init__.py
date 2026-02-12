"""
Implements flexible piecewise learning rate schedules via a builder pattern.
"""

from .builder import piecewise_schedule
from .config import PiecewiseSchedulerConfig, piecewise_scheduler_from_config
from .curves import CurveBase, CurveCosine, CurveExponential, CurveLinear, CurvePoly

__all__ = [
    "CurveBase",
    "CurveCosine",
    "CurveExponential",
    "CurveLinear",
    "CurvePoly",
    "PiecewiseSchedulerConfig",
    "piecewise_schedule",
    "piecewise_scheduler_from_config",
]
