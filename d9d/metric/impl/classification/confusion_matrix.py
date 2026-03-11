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
    """A generic metric computed from a confusion matrix.

    This class composes a processor, accumulator, and aggregator to compute
    metrics like Accuracy, Precision, Recall, or F1-Score via a confusion matrix.
    """

    def __init__(
        self,
        processor: ClassificationPredictionsProcessor,
        accumulator: ConfusionMatrixAccumulator,
        aggregator: ConfusionMatrixAggregator,
    ) -> None:
        """Constructs the ConfusionMatrixMetric object.

        Args:
            processor: The strategy used to convert raw predictions and targets into
                a format suitable for the confusion matrix.
            accumulator: The component responsible for tracking the confusion matrix
                counts (TP, FP, TN, FN) across batches.
            aggregator: The component responsible for computing the final statistic
                from the accumulated confusion matrix state.
        """
        self._processor = processor
        self._accumulator = accumulator
        self._aggregator = aggregator

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Processes and accumulates a new batch of predictions and targets.

        Args:
            preds: The raw prediction outputs.
            targets: The ground truth targets.
        """
        p, t = self._processor(preds, targets)
        self._accumulator.update(p, t)

    def sync(self, dist_context: DistributedContext) -> None:
        """Synchronizes the accumulated metric state across distributed workers.

        Args:
            dist_context: The distributed context containing synchronization details.
        """
        self._accumulator.sync()

    def compute(self) -> torch.Tensor:
        """Computes the final aggregated metric value.

        Returns:
            The calculated metric statistic tensor.
        """
        return self._aggregator(self._accumulator.state)

    def reset(self) -> None:
        """Resets the accumulated confusion matrix state to zero."""
        self._accumulator.reset()

    def to(self, device: str | torch.device | int) -> None:
        """Moves the internal metric states to the specified device.

        Args:
            device: The target target device.
        """
        self._accumulator.to(device)

    def state_dict(self) -> dict[str, Any]:
        """Retrieves the metric's current internal state dictionary.

        Returns:
            A dictionary of the accumulator's state.
        """
        return self._accumulator.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores the metric's internal state from a dictionary.

        Args:
            state_dict: The saved state dictionary to load.
        """
        self._accumulator.load_state_dict(state_dict)


class ConfusionMatrixMetricBuilder:
    """Builder for safely configuring a ConfusionMatrixMetric pipeline."""

    def __init__(self) -> None:
        """Constructs the ConfusionMatrixMetric object."""

        self._num_outputs: int | None = None
        self._processor: ClassificationPredictionsProcessor | None = None
        self._statistic: ConfusionMatrixStatistic | None = None
        self._aggregation_method: ClassificationAggregationMethod | None = None

    def _ensure_no_problem(self) -> None:
        if self._processor is not None:
            raise ValueError(
                "A problem type (binary, multiclass, multilabel) has already been configured. "
                "You cannot chain multiple problem definitions."
            )

    def _ensure_no_statistic(self) -> None:
        if self._statistic is not None:
            raise ValueError(
                "A target statistic has already been configured. "
                "You cannot evaluate multiple primary statistics in a single pipeline."
            )

    def _ensure_no_aggregation(self) -> None:
        if self._aggregation_method is not None:
            raise ValueError("An aggregation methodology has already been selected.")

    def binary(self, threshold: float = 0.5) -> Self:
        """Configures the metric for binary classification problems.

        Args:
            threshold: Value boundary for assigning positive boolean classes.

        Returns:
            The current builder instance.
        """
        self._ensure_no_problem()
        self._processor = ThresholdProcessor(threshold)
        self._num_outputs = 1
        self._aggregation_method = ClassificationAggregationMethod.MICRO
        return self

    def multiclass(self, num_classes: int, top_k: int | None = None) -> Self:
        """Configures the metric for multiclass classification problems.

        Args:
            num_classes: The total number of unique mutually-exclusive classes.
            top_k: If provided, alters the underlying evaluation to measure if the
                target falls within the top K highest probabilities.

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
        """Configures the metric for multilabel classification problems.

        Args:
            num_classes: The total number of unique independent classes.
            threshold: Value boundary for assigning positive boolean hits independently.

        Returns:
            The current builder instance.
        """
        self._ensure_no_problem()

        self._processor = ThresholdProcessor(threshold)
        self._num_outputs = num_classes
        return self

    def with_accuracy(self) -> Self:
        """Assigns conventional accuracy computations as the target statistic to evaluate.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = AccuracyStatistic()
        return self

    def with_f1(self) -> Self:
        """Assigns harmonic mean calculations (F1) as the target statistic to evaluate.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = FBetaStatistic(beta=1)
        return self

    def with_fbeta(self, beta: float) -> Self:
        """Assigns variable recall-focused FBeta score as the target statistic to evaluate.

        Args:
            beta: Emphasis coefficient towards recall impact strictly mathematically.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = FBetaStatistic(beta)
        return self

    def with_precision(self) -> Self:
        """Assigns target hit accuracy distribution (Precision) as the target statistic.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = PrecisionStatistic()
        return self

    def with_recall(self) -> Self:
        """Assigns target missing reduction distribution (Recall) as the target statistic.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = RecallStatistic()
        return self

    def with_statistic(self, statistic: ConfusionMatrixStatistic) -> Self:
        """Assigns entirely custom formulas interpreting matrix states natively.

        Args:
            statistic: Instantiated formulation protocol.

        Returns:
            The current builder instance.
        """
        self._ensure_no_statistic()
        self._statistic = statistic
        return self

    def with_aggregation(self, method: ClassificationAggregationMethod) -> Self:
        """Configures general target methodology formulas defining multi-dimensional matrices.

        Args:
            method: Constant identifier pointing to strategy options available.

        Returns:
            The current builder instance.
        """
        self._ensure_no_aggregation()
        self._aggregation_method = method
        return self

    def micro(self) -> Self:
        """Computes the metric globally by summing the confusion matrices first.

        Returns:
            The current builder instance.
        """
        return self.with_aggregation(ClassificationAggregationMethod.MICRO)

    def macro(self) -> Self:
        """Computes the metric for each class independently and finds their unweighted mean.

        Returns:
            The current builder instance.
        """
        return self.with_aggregation(ClassificationAggregationMethod.MACRO)

    def weighted(self) -> Self:
        """Computes the metric for each class independently and finds their average weighted by the true instances
            (support) for each class.

        Returns:
            The current builder instance.
        """
        return self.with_aggregation(ClassificationAggregationMethod.WEIGHTED)

    def per_class(self) -> Self:
        """Computes and returns the metric for each class independently without aggregating.

        Returns:
            The current builder instance.
        """
        return self.with_aggregation(ClassificationAggregationMethod.NONE)

    def build(self) -> ConfusionMatrixMetric:
        """Bakes pipeline configurations into a ``ConfusionMatrixMetric``.

        Returns:
            A ready-to-process configured metric wrapper instance.

        Raises:
            ValueError: If the problem type or statistic calculation is not specified.
        """
        if self._processor is None or self._num_outputs is None:
            raise ValueError("A problem type (binary, multiclass, multilabel) must be configured.")

        if self._statistic is None:
            raise ValueError("A statistic calculation strategy must be configured.")

        if self._aggregation_method is None:
            raise ValueError("Aggregation method must be configured.")

        accumulator = ConfusionMatrixAccumulator(self._num_outputs)
        aggregator = ConfusionMatrixAggregator(self._aggregation_method, self._statistic)

        return ConfusionMatrixMetric(
            processor=self._processor,
            accumulator=accumulator,
            aggregator=aggregator,
        )


def confusion_matrix_metric() -> ConfusionMatrixMetricBuilder:
    """Creates a new builder for configuring a ConfusionMatrixMetric.

    Returns:
        A fresh builder instance to begin metric pipeline configuration.
    """
    return ConfusionMatrixMetricBuilder()
