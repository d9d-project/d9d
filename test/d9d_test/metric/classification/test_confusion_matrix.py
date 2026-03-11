import pytest
import torch
from d9d.metric.impl.classification import confusion_matrix_metric

from d9d_test.metric.infra import (
    MetricCase,
    MetricParams,
    MetricStep,
    assert_metric_distributed,
    assert_metric_local,
)


@pytest.mark.local
def test_builder_exceptions():
    with pytest.raises(ValueError, match="already been configured"):
        confusion_matrix_metric().binary().multiclass(2)

    with pytest.raises(ValueError, match="already been configured"):
        confusion_matrix_metric().with_accuracy().with_f1()

    with pytest.raises(ValueError, match="already been selected"):
        confusion_matrix_metric().micro().macro()

    with pytest.raises(ValueError, match="already been selected"):
        confusion_matrix_metric().binary().micro()

    with pytest.raises(ValueError, match="calculation strategy must be configured"):
        confusion_matrix_metric().binary().build()


@pytest.mark.local
def test_execution_exceptions():
    metric = confusion_matrix_metric().multiclass(num_classes=3).with_accuracy().micro().build()

    # 1. Prediction's trailing dimension doesn't match configured num_classes
    with pytest.raises(ValueError, match="Expected last dimension of preds to equal num_classes=3"):
        metric.update(preds=torch.rand(2, 4), targets=torch.randint(0, 3, (2,)))

    # 2. Incompatible target shapes (neither match (...), (..., 1), nor (..., C))
    with pytest.raises(ValueError, match="incompatible with predictions shape"):
        # preds is (2, 3), targets is (2, 2)
        metric.update(preds=torch.rand(2, 3), targets=torch.randint(0, 3, (2, 2)))

    binary_metric = confusion_matrix_metric().binary().with_accuracy().build()

    # 3. Accumulated state shape mismatch in binary component
    with pytest.raises(ValueError, match="preds and targets must have same shape"):
        # The ThresholdProcessor fixes 1D to 2D but doesn't fix entirely broken pairs
        binary_metric.update(preds=torch.rand(4), targets=torch.rand(5))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "case",
    [
        # Case 1: Binary Per-Class Accuracy
        # Target returns a 1D tensor [value]
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().binary(threshold=0.5).with_accuracy().build(),
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    [
                        MetricParams(
                            preds=torch.tensor([0.9, 0.1, 0.4, 0.8, 0.9]), targets=torch.tensor([1, 0, 1, 0, 1])
                        )
                    ],
                    expect=torch.tensor(3 / 5),
                ),
                MetricStep(
                    [MetricParams(preds=torch.tensor([0.9, 0.1, 0.1]), targets=torch.tensor([1, 1, 0]))],
                    expect=torch.tensor(5 / 8),
                ),
            ],
        ),
        # Case 2: Multiclass Macro F1
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().multiclass(num_classes=3).with_f1().macro().build(),
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    [
                        MetricParams(
                            # Pred indices: [1, 1, 0, 2, 2, 1]
                            preds=torch.tensor(
                                [
                                    [0.1, 0.8, 0.1],
                                    [0.1, 0.7, 0.2],
                                    [0.6, 0.2, 0.2],
                                    [0.1, 0.1, 0.8],
                                    [0.2, 0.2, 0.6],
                                    [0.3, 0.5, 0.2],
                                ]
                            ),
                            targets=torch.tensor([0, 1, 2, 2, 2, 1]),
                        )
                    ],
                    # Stats per class:
                    # C0: True=1, Pred=1. TP=0, FN=1, FP=1. F1 = 0.0
                    # C1: True=2, Pred=3. TP=2, FN=0, FP=1. F1 = 4 / 5 = 0.8
                    # C2: True=3, Pred=2. TP=2, FN=1, FP=0. F1 = 4 / 5 = 0.8
                    expect=torch.tensor((0.0 + 0.8 + 0.8) / 3),
                )
            ],
        ),
        # Case 3: Multilabel Micro Accuracy
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric()
            .multilabel(num_classes=3, threshold=0.5)
            .with_accuracy()
            .micro()
            .build(),
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    [
                        MetricParams(
                            # Pred binarized: [1, 0, 0], [0, 1, 0], [0, 1, 1]
                            preds=torch.tensor([[0.8, 0.2, 0.3], [0.4, 0.9, 0.1], [0.3, 0.8, 0.7]]),
                            targets=torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]]),
                        )
                    ],
                    # Total hits: 2 + 2 + 3 = 7. Total queries: 9.
                    expect=torch.tensor(7 / 9),
                )
            ],
        ),
        # Case 4: Multiclass Weighted Recall
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().multiclass(num_classes=3).with_recall().weighted().build(),
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    [
                        MetricParams(
                            # Pred indices: [0, 1, 0, 1, 2, 2]
                            preds=torch.tensor(
                                [
                                    [0.9, 0.0, 0.0],
                                    [0.0, 0.9, 0.0],
                                    [0.9, 0.0, 0.0],
                                    [0.0, 0.9, 0.0],
                                    [0.0, 0.0, 0.9],
                                    [0.0, 0.0, 0.9],
                                ]
                            ),
                            # True labels: [0, 1, 2, 1, 1, 2]
                            targets=torch.tensor([0, 1, 2, 1, 1, 2]),
                        )
                    ],
                    expect=torch.tensor((1.0 * 1 + (2 / 3) * 3 + 0.5 * 2) / 6),
                )
            ],
        ),
        # Case 5: Multiclass Top-K Micro Accuracy
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().multiclass(num_classes=5, top_k=2).with_accuracy().build(),
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    [
                        MetricParams(
                            # Top2: [2,0], [4,1], [1,0], [4,2], [3,1]
                            preds=torch.tensor(
                                [
                                    [0.2, 0.1, 0.5, 0.1, 0.1],
                                    [0.1, 0.3, 0.1, 0.1, 0.4],
                                    [0.3, 0.4, 0.1, 0.1, 0.1],
                                    [0.1, 0.1, 0.3, 0.1, 0.4],
                                    [0.1, 0.3, 0.1, 0.4, 0.1],
                                ]
                            ),
                            targets=torch.tensor([2, 4, 1, 3, 0]),
                        )
                    ],
                    expect=torch.tensor(3 / 5),
                )
            ],
        ),
        # Case 6: Binary F-Beta
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().binary().with_fbeta(beta=2.0).build(),
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    [
                        MetricParams(
                            # Pred binarized (thresh 0.5): [1, 1, 0, 1, 0, 0]
                            preds=torch.tensor([0.9, 0.8, 0.1, 0.7, 0.2, 0.3]),
                            # Targets: [1, 0, 1, 1, 0, 0]
                            targets=torch.tensor([1, 0, 1, 1, 0, 0]),
                        )
                    ],
                    # Matrix: TP=2, FP=1, FN=1, TN=2
                    # F-Beta(2.0): (1+4)*TP / ((1+4)*TP + 4*FN + FP)
                    # 5*2 / (10 + 4*1 + 1) = 10 / 15 = 0.6666667
                    expect=torch.tensor(5 * 2 / (10 + 4 * 1 + 1)),
                )
            ],
        ),
        # Case 7: Multiclass with explicit One-Hot Encoded Targets
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().multiclass(num_classes=3).with_accuracy().per_class().build(),
            initial_expect=torch.tensor([torch.nan, torch.nan, torch.nan]),
            steps=[
                MetricStep(
                    [
                        MetricParams(
                            # Preds: C1, C1, C0
                            preds=torch.tensor([[0.1, 0.8, 0.1], [0.2, 0.7, 0.1], [0.9, 0.05, 0.05]]),
                            # Targets One-Hot: C1, C2, C0 => [0, 1, 0], [0, 0, 1], [1, 0, 0]
                            targets=torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
                        )
                    ],
                    # C0: TP=1, TN=2, FP=0, FN=0 -> Acc = 3/3
                    # C1: TP=1, TN=1, FP=1, FN=0 -> Acc = 2/3
                    # C2: TP=0, TN=2, FP=0, FN=1 -> Acc = 2/3
                    expect=torch.tensor([1.0, 2 / 3, 2 / 3]),
                )
            ],
        ),
        # Case 8: Multiclass with trailing Target Dimension (..., 1)
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().multiclass(num_classes=3).with_accuracy().per_class().build(),
            initial_expect=torch.tensor([torch.nan, torch.nan, torch.nan]),
            steps=[
                MetricStep(
                    [
                        MetricParams(
                            # Preds: C1, C2, C1, C2 (idx: 1, 2, 1, 2)
                            preds=torch.tensor(
                                [
                                    [0.1, 0.8, 0.1],
                                    [0.1, 0.1, 0.8],
                                    [0.2, 0.7, 0.1],
                                    [0.1, 0.2, 0.7],
                                ]
                            ),
                            # Targets (..., 1): [[1], [2], [0], [2]]
                            targets=torch.tensor([[1], [2], [0], [2]]),
                        )
                    ],
                    # Evaluated across N=4.
                    # C0: TP=0, TN=3, FP=0, FN=1 -> 3/4 = 0.75
                    # C1: TP=1, TN=2, FP=1, FN=0 -> 3/4 = 0.75
                    # C2: TP=2, TN=2, FP=0, FN=0 -> 4/4 = 1.0
                    expect=torch.tensor([0.75, 0.75, 1.0]),
                )
            ],
        ),
        # Case 9: Multidimensional Inputs (e.g., Sequence / Spatial Evaluation)
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().multiclass(num_classes=3).with_accuracy().micro().build(),
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    [
                        MetricParams(
                            # Shape: (Batch=2, SeqLen=2, Classes=3). Preds: [[0, 1], [2, 1]]
                            preds=torch.tensor(
                                [
                                    [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1]],
                                    [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
                                ]
                            ),
                            # Shape: (Batch=2, SeqLen=2). True: [[0, 1], [2, 0]]
                            targets=torch.tensor([[0, 1], [2, 0]]),
                        )
                    ],
                    # Each of the 4 elements evaluates 3 classes -> Total 12 tracking checks
                    # C0 True:(1,0,0,1) Pred:(1,0,0,0) -> TP=1, TN=2, FN=1, FP=0 (Hits=3)
                    # C1 True:(0,1,0,0) Pred:(0,1,0,1) -> TP=1, TN=2, FN=0, FP=1 (Hits=3)
                    # C2 True:(0,0,1,0) Pred:(0,0,1,0) -> TP=1, TN=3, FN=0, FP=0 (Hits=4)
                    # Total hits = 3 + 3 + 4 = 10. Accuracy = 10 / 12
                    expect=torch.tensor(10 / 12),
                )
            ],
        ),
        # Case 10: Extensibility - Custom Statistic Calculation
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric()
            .multiclass(num_classes=3)
            .with_statistic(lambda m: m.tp.float())
            .per_class()
            .build(),
            initial_expect=torch.tensor([0.0, 0.0, 0.0]),
            steps=[
                MetricStep(
                    [
                        MetricParams(
                            # Preds: C0, C0, C2
                            preds=torch.tensor([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.0, 0.1, 0.9]]),
                            # Targets: C0, C0, C2
                            targets=torch.tensor([0, 0, 2]),
                        )
                    ],
                    # Returning raw True Positives per Class directly: C0(2), C1(0), C2(1)
                    expect=torch.tensor([2.0, 0.0, 1.0]),
                )
            ],
        ),
        # Case 11: Division by Zero Handling (Standard PyTorch returns NaN)
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().binary().with_precision().build(),
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    [
                        # No positive predictions present (Preds thresholding to all 0s)
                        # TP = 0, FP = 0. Precision = 0 / 0 = NaN
                        MetricParams(preds=torch.tensor([0.1, 0.2]), targets=torch.tensor([1, 0]))
                    ],
                    expect=torch.tensor(torch.nan),
                )
            ],
        ),
    ],
)
@pytest.mark.local
def test_local(case: MetricCase, device: str):
    assert_metric_local(case, device)


@pytest.mark.parametrize(
    "case",
    [
        # Distributed Case 1: Binary Micro Precision
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().binary(0.5).with_precision().build(),
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    params_per_rank=[
                        # Rank 0: TP=2, FP=1
                        MetricParams(preds=torch.tensor([0.9, 0.9, 0.9]), targets=torch.tensor([1, 1, 0])),
                        # Rank 1: TP=2, FP=0
                        MetricParams(preds=torch.tensor([0.9, 0.9, 0.1]), targets=torch.tensor([1, 1, 0])),
                        # Ranks 2-7: TP=0, FP=0
                        *[
                            MetricParams(preds=torch.tensor([0.1, 0.1, 0.1]), targets=torch.tensor([0, 0, 0]))
                            for _ in range(6)
                        ],
                    ],
                    # Global: TP=4, FP=1. Precision = 4/(4+1) = 0.8
                    expect=torch.tensor(0.8),
                ),
                MetricStep(
                    params_per_rank=[
                        # Rank 0: TP=3, FP=1 (adds to total)
                        MetricParams(preds=torch.tensor([0.9, 0.9, 0.9, 0.9]), targets=torch.tensor([1, 1, 1, 0])),
                        # all others: neutral hits
                        *[MetricParams(preds=torch.tensor([0.1]), targets=torch.tensor([0])) for _ in range(7)],
                    ],
                    # Ongoing total: TP=4+3=7, FP=1+1=2. Precision = 7/9 = 0.7777778
                    expect=torch.tensor(0.7777778),
                ),
            ],
        ),
        # Distributed Case 2: Multiclass Top-K Micro Accuracy
        MetricCase(
            factory_fn=lambda: confusion_matrix_metric().multiclass(num_classes=4, top_k=2).with_accuracy().build(),
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    params_per_rank=[
                        # Rank 0: target 2 in top-2 [2, 1] => Hit
                        MetricParams(preds=torch.tensor([[0.1, 0.8, 0.9, 0.0]]), targets=torch.tensor([2])),
                        # Ranks 1-2: target 1 in top-2 [2, 1] => Hits (x2)
                        MetricParams(preds=torch.tensor([[0.1, 0.8, 0.9, 0.0]]), targets=torch.tensor([1])),
                        MetricParams(preds=torch.tensor([[0.1, 0.8, 0.9, 0.0]]), targets=torch.tensor([1])),
                        # Ranks 3-4: target 2 in top-2 [2, 1] => Hits (x2)
                        MetricParams(preds=torch.tensor([[0.1, 0.8, 0.9, 0.0]]), targets=torch.tensor([2])),
                        MetricParams(preds=torch.tensor([[0.1, 0.8, 0.9, 0.0]]), targets=torch.tensor([2])),
                        # Ranks 5-7: target 0 in top-2 [2, 1] => Misses (x3)
                        MetricParams(preds=torch.tensor([[0.1, 0.8, 0.9, 0.0]]), targets=torch.tensor([0])),
                        MetricParams(preds=torch.tensor([[0.1, 0.8, 0.9, 0.0]]), targets=torch.tensor([0])),
                        MetricParams(preds=torch.tensor([[0.1, 0.8, 0.9, 0.0]]), targets=torch.tensor([0])),
                    ],
                    # Global: 5 Hits out of 8 targets => Accuracy = 5/8 = 0.625
                    expect=torch.tensor(0.625),
                )
            ],
        ),
    ],
)
@pytest.mark.distributed
def test_distributed(dist_ctx_factory, case: MetricCase):
    assert_metric_distributed(dist_ctx_factory, case)
