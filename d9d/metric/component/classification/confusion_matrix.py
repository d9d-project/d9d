import dataclasses

import torch


@dataclasses.dataclass(kw_only=True, slots=True)
class ConfusionMatrix:
    """Represents a confusion matrix for classification evaluation.

    Attributes:
        tp: The count of true positives.
        fp: The count of false positives.
        tn: The count of true negatives.
        fn: The count of false negatives.
    """

    tp: torch.Tensor
    fp: torch.Tensor
    tn: torch.Tensor
    fn: torch.Tensor
