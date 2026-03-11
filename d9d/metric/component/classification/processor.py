from typing import Protocol

import torch
import torch.nn.functional as F


class ClassificationPredictionsProcessor(Protocol):
    """Protocol for processing classification predictions and targets into a format suitable for evaluation."""

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms raw predictions and targets into standardized tensors.

        Args:
            preds: The raw prediction outputs from the model. Expected shape depends
                on the specific implementation.
            targets: The ground truth targets. Expected shape depends on the
                specific implementation.

        Returns:
            A tuple containing the processed predictions and processed targets tensors.
        """
        ...


class TopKProcessor(ClassificationPredictionsProcessor):
    """Processes classification predictions to evaluate top-k accuracy."""

    def __init__(self, k: int) -> None:
        """Constructs the TopKProcessor object.

        Args:
            k: The number of highest probability predictions to consider for a hit.
        """
        self._k = k

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Checks if the true target is within the top-k predicted logits.

        Args:
            preds: The raw prediction logits or probabilities of shape (..., C), where C is the number
                of classes.
            targets: The ground truth class indices of shape (...), matching the
                leading dimensions of the predictions.

        Returns:
            A tuple containing a boolean hit/miss tensor of shape (..., 1) and a dummy
            target tensor of ones of matching shape (..., 1).
        """
        # Get top-k indices over the last dimension: (..., k)
        _, topk_indices = torch.topk(preds, self._k, dim=-1)

        # Check if target is in top-k
        # targets.unsqueeze(-1) turns (...) into (..., 1), enabling automatic broadcasting
        is_hit = (topk_indices == targets.unsqueeze(-1)).any(dim=-1, keepdim=True).long()

        # dummy_target is always 1 (Hit) because we want to compare Prediction (Hit/Miss) vs Ideal (Hit)
        dummy_target = torch.ones_like(is_hit)

        return is_hit, dummy_target


class OneHotProcessor(ClassificationPredictionsProcessor):
    """Processes predictions and targets by computing argmax and converting them into one-hot format."""

    def __init__(self, num_classes: int) -> None:
        """Constructs the OneHotProcessor object.

        Args:
            num_classes: The total number of unique classes for one-hot encoding.
        """
        self._num_classes = num_classes

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts categorical predictions and targets to one-hot encoded format.

        Args:
            preds: The raw prediction logits or probabilities of shape (..., C), where C is the number
                of classes.
            targets: The ground truth class indices of shape (...), (..., 1), or already
                one-hot encoded targets of shape (..., C).

        Returns:
            A tuple containing one-hot encoded predictions and targets tensors, both of
                shape (..., C).

        Raises:
            ValueError: If the targets tensor shape is incompatible with the predictions.
        """

        if preds.shape[-1] != self._num_classes:
            raise ValueError(
                f"Expected last dimension of preds to equal num_classes={self._num_classes}, got {preds.shape[-1]}"
            )

        preds_indices = torch.argmax(preds, dim=-1)
        preds_one_hot = F.one_hot(preds_indices, num_classes=self._num_classes).long()

        if targets.shape == preds.shape:
            # Targets are already (..., C)
            targets_one_hot = targets.long()
        elif targets.shape == preds.shape[:-1]:
            # Targets are (...) representing integer class labels
            targets_one_hot = F.one_hot(targets.long(), num_classes=self._num_classes).long()
        elif targets.shape == (*preds.shape[:-1], 1):
            # Targets are (..., 1) representing integer class labels with explicit trailing dim
            targets_one_hot = F.one_hot(targets.squeeze(-1).long(), num_classes=self._num_classes).float()
        else:
            raise ValueError(
                f"Targets shape {targets.shape} is incompatible with predictions shape {preds.shape}."
                f"Expected shape to be (...), (..., 1), or (..., C)."
            )

        return preds_one_hot, targets_one_hot


class ThresholdProcessor(ClassificationPredictionsProcessor):
    """Processes probabilistic predictions by applying a binary threshold."""

    def __init__(self, threshold: float) -> None:
        """Constructs the ThresholdProcessor object.

        Args:
            threshold: The boundary value above which a prediction is considered positive.
        """
        self._threshold = threshold

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Binarizes predictions based on the configured threshold.

        Args:
            preds: The prediction probabilities of shape (..., C) or 1D shape (N,) containing
                values, typically in the range [0, 1].
            targets: The ground truth targets of shape (..., C) or 1D shape (N,).

        Returns:
            A tuple containing binarized predictions and float targets tensors, both
            ensured to have at least 2 dimensions, yielding patterns like (..., C) or (N, 1).
        """
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)

        binary_preds = (preds > self._threshold).float()
        return binary_preds, targets.float()
