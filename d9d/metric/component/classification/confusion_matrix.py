import dataclasses

import torch


@dataclasses.dataclass(kw_only=True, slots=True)
class ConfusionMatrix:
    """Represents a confusion matrix for classification evaluation.

    Attributes:
        tp: Tensor containing the count of true positives.
        fp: Tensor containing the count of false positives.
        tn: Tensor containing the count of true negatives.
        fn: Tensor containing the count of false negatives.
    """

    tp: torch.Tensor
    fp: torch.Tensor
    tn: torch.Tensor
    fn: torch.Tensor
