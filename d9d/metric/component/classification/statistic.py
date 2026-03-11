from typing import Protocol

import torch

from .confusion_matrix import ConfusionMatrix


class ConfusionMatrixStatistic(Protocol):
    """Protocol for computing statistics from a confusion matrix."""

    def __call__(self, matrix: ConfusionMatrix) -> torch.Tensor:
        """Computes a statistic from the given confusion matrix.

        Args:
            matrix: The confusion matrix to compute the statistic from.

        Returns:
            The computed statistic.
        """
        ...


class PrecisionStatistic(ConfusionMatrixStatistic):
    """Computes the precision statistic from a confusion matrix."""

    def __call__(self, matrix: ConfusionMatrix) -> torch.Tensor:
        return matrix.tp / (matrix.tp + matrix.fp)


class RecallStatistic(ConfusionMatrixStatistic):
    """Computes the recall statistic from a confusion matrix."""

    def __call__(self, matrix: ConfusionMatrix) -> torch.Tensor:
        return matrix.tp / (matrix.tp + matrix.fn)


class FBetaStatistic(ConfusionMatrixStatistic):
    """Computes the F-beta score from a confusion matrix."""

    def __init__(self, beta: float) -> None:
        """Constructs the FBetaStatistic object.

        Args:
            beta: The beta parameter determining the weight of recall mathematically
                relative to precision.
        """
        self._beta_sq = beta**2

    def __call__(self, matrix: ConfusionMatrix) -> torch.Tensor:
        numerator = (1 + self._beta_sq) * matrix.tp
        denominator = (1 + self._beta_sq) * matrix.tp + self._beta_sq * matrix.fn + matrix.fp
        return numerator / denominator


class AccuracyStatistic(ConfusionMatrixStatistic):
    """Computes the accuracy statistic from a confusion matrix."""

    def __call__(self, matrix: ConfusionMatrix) -> torch.Tensor:
        return (matrix.tp + matrix.tn) / (matrix.tp + matrix.tn + matrix.fp + matrix.fn)
