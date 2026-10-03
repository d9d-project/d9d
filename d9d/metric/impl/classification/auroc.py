from typing import Any

import torch

from d9d.core.dist_context import DistributedContext
from d9d.metric import Metric
from d9d.metric.component import MetricAccumulator


def _compute_histogram_auroc(pos_hist: torch.Tensor, neg_hist: torch.Tensor) -> torch.Tensor:
    """Computes AUROC from positive and negative histograms with the trapezoidal rule.

    It approximates ``P(X > Y) + 0.5 * P(X = Y)``, where ``X`` is the score of a positive sample and ``Y`` is the
    score of a negative sample.

    Args:
        pos_hist: Histogram counts for positive samples.
        neg_hist: Histogram counts for negative samples.

    Returns:
        The area under the curve, or 0.5 if one class is absent.
    """
    total_pos = pos_hist.sum()
    total_neg = neg_hist.sum()

    # AUROC is undefined if one class is absent.
    if total_pos <= 0 or total_neg <= 0:
        return pos_hist.new_tensor(0.5)

    cum_pos = pos_hist.cumsum(dim=0)
    acc_pos = total_pos - cum_pos

    area = ((0.5 * neg_hist * pos_hist) + (neg_hist * acc_pos)).sum()
    denominator = total_pos * total_neg

    raw_auroc = area / denominator

    # The early return above already handles a missing class; this guard also returns 0.5 instead of inf/nan.
    is_valid = (total_pos > 0) & (total_neg > 0)
    return torch.where(is_valid, raw_auroc, pos_hist.new_tensor(0.5))


class BinaryAUROCMetric(Metric[torch.Tensor]):
    """Computes an approximate AUROC for binary classification from histograms.

    Exact AUROC needs all predictions to sort and rank them. This metric keeps two histograms of predicted
    probabilities instead, one for positive and one for negative samples. It estimates the area with the
    trapezoidal rule, so memory does not grow with the number of samples. If one class is absent, the result
    is 0.5.

    References:
        Albakour et al., [Fast and memory efficient AUC-ROC approximation for Stream Learning](https://www.researchgate.net/publication/353020448_Fast_and_memory_efficient_AUC-ROC_approximation_for_Stream_Learning),
        2021.
    """

    def __init__(self, num_bins: int = 10000):
        """Constructs the ``BinaryAUROCMetric`` object.

        Args:
            num_bins: Number of histogram bins. More bins use more memory and give a more accurate result.
        """
        self._num_bins = num_bins
        self._device: str | torch.device | int = "cpu"

        shape = (num_bins,)
        self._pos_hist = MetricAccumulator(torch.zeros(shape, dtype=torch.float32))
        self._neg_hist = MetricAccumulator(torch.zeros(shape, dtype=torch.float32))

    def update(self, probs: torch.Tensor, labels: torch.Tensor):
        """Adds a batch of predictions to the histograms.

        Args:
            probs: Predicted probabilities in ``[0, 1]``. Any shape.
            labels: Ground truth binary labels, with as many elements as ``probs``.

        Raises:
            ValueError: If ``probs`` and ``labels`` have a different number of elements.
        """
        probs = probs.reshape(-1)
        labels = labels.reshape(-1)

        if probs.numel() != labels.numel():
            raise ValueError(
                f"probs ({probs.numel()} elements) and labels ({labels.numel()} elements) "
                "must have the same number of elements."
            )

        bins = (probs * self._num_bins).long().clamp(0, self._num_bins - 1)
        pos_batch = torch.zeros(self._num_bins, device=self._device, dtype=torch.float32)
        neg_batch = torch.zeros(self._num_bins, device=self._device, dtype=torch.float32)
        pos_batch.index_add_(0, bins, labels.float())
        neg_batch.index_add_(0, bins, (1 - labels).float())

        self._pos_hist.update(pos_batch)
        self._neg_hist.update(neg_batch)

    def sync(self, dist_context: DistributedContext):
        self._pos_hist.sync()
        self._neg_hist.sync()

    def compute(self) -> torch.Tensor:
        pos_hist = self._pos_hist.value
        neg_hist = self._neg_hist.value
        return _compute_histogram_auroc(pos_hist, neg_hist)

    def reset(self):
        self._pos_hist.reset()
        self._neg_hist.reset()

    def to(self, device: str | torch.device | int):
        self._device = device
        self._pos_hist.to(device)
        self._neg_hist.to(device)

    def state_dict(self) -> dict[str, Any]:
        return {"pos_hist": self._pos_hist.state_dict(), "neg_hist": self._neg_hist.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._pos_hist.load_state_dict(state_dict["pos_hist"])
        self._neg_hist.load_state_dict(state_dict["neg_hist"])
