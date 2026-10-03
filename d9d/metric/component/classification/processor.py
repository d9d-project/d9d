from typing import Protocol

import torch
import torch.nn.functional as F


class ClassificationPredictionsProcessor(Protocol):
    """Protocol for converting classification predictions and targets into binary tensors for evaluation."""

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts raw predictions and targets into binary tensors of the same shape.

        Args:
            preds: The raw model predictions. The expected shape depends on the implementation.
            targets: The ground truth targets. The expected shape depends on the implementation.

        Returns:
            The processed predictions and targets. Shape: ``(..., num_outputs)``.
        """
        ...


class TopKProcessor(ClassificationPredictionsProcessor):
    """Processes classification predictions to evaluate top-k accuracy."""

    def __init__(self, k: int) -> None:
        """Constructs the ``TopKProcessor`` object.

        Args:
            k: The number of highest-scored classes that count as a hit.
        """
        self._k = k

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Checks whether the true class is among the top-k predicted classes.

        Args:
            preds: The prediction logits or probabilities. Shape: ``(..., num_classes)``.
            targets: The ground truth class indices. Shape: ``(...)``.

        Returns:
            A hit (1) or miss (0) tensor and a target tensor of ones. Shape: ``(..., 1)``.
        """
        # Shape: (..., k).
        _, topk_indices = torch.topk(preds, self._k, dim=-1)

        is_hit = (topk_indices == targets.unsqueeze(-1)).any(dim=-1, keepdim=True).long()

        # The ideal outcome is always a hit, so the target is all ones.
        dummy_target = torch.ones_like(is_hit)

        return is_hit, dummy_target


class OneHotProcessor(ClassificationPredictionsProcessor):
    """Converts the argmax of predictions and the targets into one-hot tensors."""

    def __init__(self, num_classes: int) -> None:
        """Constructs the ``OneHotProcessor`` object.

        Args:
            num_classes: The number of classes.
        """
        self._num_classes = num_classes

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts predictions and targets to one-hot tensors.

        Args:
            preds: The prediction logits or probabilities. Shape: ``(..., num_classes)``.
            targets: The ground truth class indices with shape ``(...)`` or ``(..., 1)``, or one-hot targets with
                shape ``(..., num_classes)``.

        Returns:
            The one-hot predictions and targets. Shape: ``(..., num_classes)``.

        Raises:
            ValueError: If the last dimension of ``preds`` is not ``num_classes``, or if the shape of ``targets``
                does not match ``preds``.
        """
        if preds.shape[-1] != self._num_classes:
            raise ValueError(
                f"The last dimension of preds ({preds.shape[-1]}) must equal num_classes ({self._num_classes})."
            )

        preds_indices = torch.argmax(preds, dim=-1)
        preds_one_hot = F.one_hot(preds_indices, num_classes=self._num_classes).long()

        if targets.shape == preds.shape:
            # Targets are already one-hot: (..., num_classes).
            targets_one_hot = targets.long()
        elif targets.shape == preds.shape[:-1]:
            # Targets are class indices: (...).
            targets_one_hot = F.one_hot(targets.long(), num_classes=self._num_classes).long()
        elif targets.shape == (*preds.shape[:-1], 1):
            # Targets are class indices with a trailing dimension: (..., 1).
            targets_one_hot = F.one_hot(targets.squeeze(-1).long(), num_classes=self._num_classes).float()
        else:
            raise ValueError(
                f"Targets shape ({tuple(targets.shape)}) is incompatible with predictions shape "
                f"({tuple(preds.shape)}). Targets must have shape (...), (..., 1) or (..., num_classes)."
            )

        return preds_one_hot, targets_one_hot


class ThresholdProcessor(ClassificationPredictionsProcessor):
    """Binarizes probability predictions with a threshold."""

    def __init__(self, threshold: float) -> None:
        """Constructs the ``ThresholdProcessor`` object.

        Args:
            threshold: Predictions strictly above this value are positive.
        """
        self._threshold = threshold

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Binarizes predictions with the configured threshold.

        Args:
            preds: The predictions, typically probabilities in ``[0, 1]``. Shape: ``(..., num_classes)`` or
                ``(num_samples,)``.
            targets: The binary ground truth targets. Shape: ``(..., num_classes)`` or ``(num_samples,)``.

        Returns:
            The binary predictions and the targets as float tensors. 1D inputs get a trailing dimension.
            Shape: ``(..., num_classes)`` or ``(num_samples, 1)``.
        """
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)

        binary_preds = (preds > self._threshold).float()
        return binary_preds, targets.float()
