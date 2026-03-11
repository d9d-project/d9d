from enum import StrEnum

import torch

from .confusion_matrix import ConfusionMatrix
from .statistic import ConfusionMatrixStatistic


class ClassificationAggregationMethod(StrEnum):
    """Defines methods for aggregating metrics across multiple classes.

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
    """Aggregates a confusion matrix state into a single statistic tensor.

    This class evaluates a given statistic across a multiclass confusion matrix
    using a specified aggregation method.
    """

    def __init__(self, method: ClassificationAggregationMethod, statistic: ConfusionMatrixStatistic) -> None:
        """Constructs the ConfusionMatrixAggregator object.

        Args:
            method: The methodology used to aggregate the matrices or statistics.
            statistic: The protocol or callable responsible for computing the statistic
                from the confusion matrix.
        """
        self._method = method
        self._statistic = statistic

    def __call__(self, matrix: ConfusionMatrix) -> torch.Tensor:
        """Aggregates the given confusion matrix and computes the target statistic.

        Args:
            matrix: The accumulated confusion matrix state containing class counts. Shape of each its cell is (C,)

        Returns:
            The computed statistic tensor. Output shape depends on the aggregation method
                (scalar for MICRO, MACRO, and WEIGHTED; 1D tensor for NONE).

        Raises:
            ValueError: If an unknown aggregation method was specified.
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
                raise ValueError(f"Unknown aggregation method: {self._method}")
