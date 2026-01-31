"""
Implements flexible piecewise learning rate schedules via a builder pattern.
"""

from .builder import piecewise_schedule
from .curves import CurveBase, CurveCosine, CurveExponential, CurveLinear, CurvePoly

__all__ = [
    "CurveBase",
    "CurveCosine",
    "CurveExponential",
    "CurveLinear",
    "CurvePoly",
    "piecewise_schedule",
]
