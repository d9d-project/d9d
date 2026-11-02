"""Metrics that aggregate values, e.g. a sum or a weighted mean."""

from .mean import WeightedMeanMetric
from .sum import SumMetric

__all__ = [
    "SumMetric",
    "WeightedMeanMetric",
]
