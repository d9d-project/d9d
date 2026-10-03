from enum import StrEnum

import torch

from .confusion_matrix import ConfusionMatrix
from .statistic import ConfusionMatrixStatistic


class ClassificationAggregationMethod(StrEnum):
    """Methods for aggregating metrics across classes.

    Attributes:
        MICRO: Computes the metric globally by summing the confusion matrices first.
        MACRO: Computes the metric for each class independently and finds their unweighted mean.
        WEIGHTED: Computes the metric for each class independently and finds their average
            weighted by the true instances (support) for each class.
        NONE: Computes and returns the metric for each class independently without aggregating.
    """

    MICRO = "micro"
    MACRO = "macro"
    WEIGHTED = "weighted"
    NONE = "none"


class ConfusionMatrixAggregator:
    """Aggregator that computes a statistic from a confusion matrix and aggregates it across classes."""

    def __init__(self, method: ClassificationAggregationMethod, statistic: ConfusionMatrixStatistic) -> None:
        """Constructs the ``ConfusionMatrixAggregator`` object.

        Args:
            method: The aggregation method across classes.
            statistic: The statistic to compute from the confusion matrix.
        """
        self._method = method
        self._statistic = statistic

    def __call__(self, matrix: ConfusionMatrix) -> torch.Tensor:
        """Computes the statistic and aggregates it with the configured method.

        Args:
            matrix: The accumulated confusion matrix. Each count has shape ``(num_outputs,)``.

        Returns:
            The statistic. A scalar for ``MICRO``, ``MACRO`` and ``WEIGHTED``. Shape: ``(num_outputs,)`` for
            ``NONE``.

        Raises:
            ValueError: If the aggregation method is unknown.
        """
        match self._method:
            case ClassificationAggregationMethod.MICRO:
                global_cm = ConfusionMatrix(
                    tp=matrix.tp.sum(),
                    fp=matrix.fp.sum(),
                    tn=matrix.tn.sum(),
                    fn=matrix.fn.sum(),
                )
                return self._statistic(global_cm)

            case ClassificationAggregationMethod.MACRO:
                scores = self._statistic(matrix)
                return scores.mean()

            case ClassificationAggregationMethod.WEIGHTED:
                scores = self._statistic(matrix)
                supports = matrix.tp + matrix.fn

                return (scores * supports).sum() / supports.sum()

            case ClassificationAggregationMethod.NONE:
                return self._statistic(matrix)

            case _:
                raise ValueError(f"Unknown aggregation method ({self._method}).")
