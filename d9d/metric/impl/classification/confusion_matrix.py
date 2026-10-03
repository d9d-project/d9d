from typing import Any, Self

import torch

from d9d.core.dist_context import DistributedContext
from d9d.metric import Metric
from d9d.metric.component.classification import (
    AccuracyStatistic,
    ClassificationAggregationMethod,
    ClassificationPredictionsProcessor,
    ConfusionMatrixAccumulator,
    ConfusionMatrixAggregator,
    ConfusionMatrixStatistic,
    FBetaStatistic,
    OneHotProcessor,
    PrecisionStatistic,
    RecallStatistic,
    ThresholdProcessor,
    TopKProcessor,
)


class ConfusionMatrixMetric(Metric[torch.Tensor]):
    """Computes a statistic, such as accuracy, precision, recall or F1 score, from a confusion matrix.

    Use ``confusion_matrix_metric()`` to build it.
    """

    def __init__(
        self,
        processor: ClassificationPredictionsProcessor,
        accumulator: ConfusionMatrixAccumulator,
        aggregator: ConfusionMatrixAggregator,
    ) -> None:
        """Constructs the ``ConfusionMatrixMetric`` object.

        Args:
            processor: Converts raw predictions and targets into binary tensors.
            accumulator: Tracks the confusion matrix counts (TP, FP, TN, FN) across batches.
            aggregator: Computes the final statistic from the accumulated confusion matrix.
        """
        self._processor = processor
        self._accumulator = accumulator
        self._aggregator = aggregator

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Processes and accumulates a batch of predictions and targets.

        Args:
            preds: The raw predictions. The expected shape depends on the problem type.
            targets: The ground truth targets. The expected shape depends on the problem type.
        """
        p, t = self._processor(preds, targets)
        self._accumulator.update(p, t)

    def sync(self, dist_context: DistributedContext) -> None:
        self._accumulator.sync()

    def compute(self) -> torch.Tensor:
        return self._aggregator(self._accumulator.state)

    def reset(self) -> None:
        self._accumulator.reset()

    def to(self, device: str | torch.device | int) -> None:
        self._accumulator.to(device)

    def state_dict(self) -> dict[str, Any]:
        return self._accumulator.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._accumulator.load_state_dict(state_dict)


class ConfusionMatrixMetricBuilder:
    """Builds a ``ConfusionMatrixMetric`` step by step.

    Choose one problem type, one statistic and, unless the problem type sets it, one aggregation method. Then call
    ``build()``.
    """

    def __init__(self) -> None:
        """Constructs the ``ConfusionMatrixMetricBuilder`` object."""
        self._num_outputs: int | None = None
        self._processor: ClassificationPredictionsProcessor | None = None
        self._statistic: ConfusionMatrixStatistic | None = None
        self._aggregation_method: ClassificationAggregationMethod | None = None

    def _ensure_no_problem(self) -> None:
        if self._processor is not None:
            raise ValueError(
                "A problem type (binary, multiclass, multilabel) has already been configured. "
                "Call only one of binary(), multiclass() or multilabel()."
            )

    def _ensure_no_statistic(self) -> None:
        if self._statistic is not None:
            raise ValueError(
                "A statistic has already been configured. A metric computes a single statistic, "
                "so call only one with_*() method."
            )

    def _ensure_no_aggregation(self) -> None:
        if self._aggregation_method is not None:
            raise ValueError(
                "An aggregation method has already been selected. "
                "Note that binary() and multiclass() with top_k select micro aggregation themselves."
            )

    def binary(self, threshold: float = 0.5) -> Self:
        """Configures the metric for binary classification.

        It also selects micro aggregation.

        Args:
            threshold: Predictions strictly above this value are positive.

        Returns:
            The current builder instance.
        """
        self._ensure_no_problem()
        self._processor = ThresholdProcessor(threshold)
        self._num_outputs = 1
        self._aggregation_method = ClassificationAggregationMethod.MICRO
        return self

    def multiclass(self, num_classes: int, top_k: int | None = None) -> Self:
        """Configures the metric for multiclass classification.

        Args:
            num_classes: The number of mutually exclusive classes.
            top_k: If set, a prediction counts as correct when the target is among the ``top_k`` highest scores.
                This also selects micro aggregation.

        Returns:
            The current builder instance.
        """
        self._ensure_no_problem()

        if top_k is not None:
            self._processor = TopKProcessor(top_k)
            self._num_outputs = 1
            self._aggregation_method = ClassificationAggregationMethod.MICRO
        else:
            self._processor = OneHotProcessor(num_classes)
            self._num_outputs = num_classes

        return self

    def multilabel(self, num_classes: int, threshold: float = 0.5) -> Self:
        """Configures the metric for multilabel classification.

        Args:
            num_classes: The number of independent classes.
            threshold: Predictions strictly above this value are positive, for each class independently.

        Returns:
            The current builder instance.
        """
        self._ensure_no_problem()

        self._processor = ThresholdProcessor(threshold)
        self._num_outputs = num_classes
        return self

    def with_accuracy(self) -> Self:
        """Selects accuracy as the statistic.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = AccuracyStatistic()
        return self

    def with_f1(self) -> Self:
        """Selects the F1 score as the statistic.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = FBetaStatistic(beta=1)
        return self

    def with_fbeta(self, beta: float) -> Self:
        """Selects the F-beta score as the statistic.

        Args:
            beta: The weight of recall relative to precision.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = FBetaStatistic(beta)
        return self

    def with_precision(self) -> Self:
        """Selects precision as the statistic.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = PrecisionStatistic()
        return self

    def with_recall(self) -> Self:
        """Selects recall as the statistic.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = RecallStatistic()
        return self

    def with_statistic(self, statistic: ConfusionMatrixStatistic) -> Self:
        """Selects a custom statistic.

        Args:
            statistic: The statistic to compute from the confusion matrix.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = statistic
        return self

    def with_aggregation(self, method: ClassificationAggregationMethod) -> Self:
        """Selects how the statistic is aggregated across classes.

        Args:
            method: The aggregation method.

        Returns:
            The current builder instance.
        """
        self._ensure_no_aggregation()
        self._aggregation_method = method
        return self

    def micro(self) -> Self:
        """Selects micro aggregation: the metric is computed from the confusion matrix summed over classes.

        Returns:
            The current builder instance.
        """
        return self.with_aggregation(ClassificationAggregationMethod.MICRO)

    def macro(self) -> Self:
        """Selects macro aggregation: the unweighted mean of the per-class metrics.

        Returns:
            The current builder instance.
        """
        return self.with_aggregation(ClassificationAggregationMethod.MACRO)

    def weighted(self) -> Self:
        """Selects weighted aggregation: the mean of the per-class metrics, weighted by class support.

        The support of a class is its number of true instances.

        Returns:
            The current builder instance.
        """
        return self.with_aggregation(ClassificationAggregationMethod.WEIGHTED)

    def per_class(self) -> Self:
        """Selects no aggregation: the metric is returned for each class.

        Returns:
            The current builder instance.
        """
        return self.with_aggregation(ClassificationAggregationMethod.NONE)

    def build(self) -> ConfusionMatrixMetric:
        """Builds the configured ``ConfusionMatrixMetric``.

        Returns:
            The configured metric.

        Raises:
            ValueError: If the problem type, the statistic or the aggregation method is not configured.
        """
        if self._processor is None or self._num_outputs is None:
            raise ValueError(
                "A problem type must be configured. Call binary(), multiclass() or multilabel() before build()."
            )

        if self._statistic is None:
            raise ValueError("A statistic must be configured. Call one of the with_*() methods before build().")

        if self._aggregation_method is None:
            raise ValueError(
                "An aggregation method must be configured. "
                "Call micro(), macro(), weighted(), per_class() or with_aggregation() before build()."
            )

        accumulator = ConfusionMatrixAccumulator(self._num_outputs)
        aggregator = ConfusionMatrixAggregator(self._aggregation_method, self._statistic)

        return ConfusionMatrixMetric(
            processor=self._processor,
            accumulator=accumulator,
            aggregator=aggregator,
        )


def confusion_matrix_metric() -> ConfusionMatrixMetricBuilder:
    """Creates a builder for a ``ConfusionMatrixMetric``.

    Returns:
        A new builder.
    """
    return ConfusionMatrixMetricBuilder()
