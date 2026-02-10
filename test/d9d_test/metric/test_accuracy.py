from functools import partial

import pytest
import torch
from d9d.metric.impl import BinaryAccuracyMetric

from d9d_test.metric.infra import (
    MetricCase,
    MetricParams,
    MetricStep,
    assert_metric_distributed,
    assert_metric_local,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "case",
    [
        # Case: Default Threshold (0.5), Simple 1D Tensors
        MetricCase(
            factory_fn=BinaryAccuracyMetric,
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep(
                    # Probs: [0.1 (0), 0.9 (1), 0.6 (1), 0.4 (0)]
                    # Labels: [0,       1,       1,       0      ] -> All Correct
                    [MetricParams(
                        torch.tensor([0.1, 0.9, 0.6, 0.4]),
                        torch.tensor([0, 1, 1, 0])
                    )],
                    expect=torch.tensor(1.0)
                ),
                MetricStep(
                    # Update with some wrong predictions
                    # Probs: [0.8 (1), 0.2 (0)]
                    # Labels: [0,       1      ] -> All Wrong
                    # Total State: 4 correct, 2 wrong. 4/6 = 0.6667
                    [MetricParams(
                        torch.tensor([0.8, 0.2]),
                        torch.tensor([0, 1])
                    )],
                    expect=torch.tensor(4.0 / 6.0)
                ),
            ]
        ),
        # Case: Custom Threshold (0.8)
        MetricCase(
            factory_fn=partial(BinaryAccuracyMetric, threshold=0.8),
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep(
                    # Probs: [0.7 (0), 0.9 (1)]
                    # Labels: [1,       1      ]
                    # 0.7 < 0.8 -> Pred 0 vs Label 1 -> Wrong
                    # 0.9 >= 0.8 -> Pred 1 vs Label 1 -> Correct
                    # Acc: 1/2 = 0.5
                    [MetricParams(
                        torch.tensor([0.7, 0.9]),
                        torch.tensor([1, 1])
                    )],
                    expect=torch.tensor(0.5)
                ),
            ]
        ),
        # Case: Multidimensional Tensors (Batch handling)
        MetricCase(
            factory_fn=BinaryAccuracyMetric,
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep(
                    # 2x2 Input
                    # Preds (thresh 0.5): [[0, 1], [1, 0]]
                    # Labels:             [[0, 1], [0, 1]]
                    # Correct: (0==0), (1==1), (1!=0), (0!=1) -> 2 Correct, 2 Wrong
                    [MetricParams(
                        torch.tensor([[0.1, 0.8], [0.6, 0.4]]),
                        torch.tensor([[0, 1], [0, 1]])
                    )],
                    expect=torch.tensor(0.5)
                )
            ]
        ),
        # Case: Empty/Zero-sized tensors (Edge case logic)
        MetricCase(
            factory_fn=BinaryAccuracyMetric,
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep(
                    # Update with empty tensors -> No change
                    [MetricParams(torch.tensor([]), torch.tensor([]))],
                    expect=torch.tensor(0.0)
                ),
                MetricStep(
                    # First valid update: 1 Correct
                    [MetricParams(torch.tensor([0.9]), torch.tensor([1]))],
                    expect=torch.tensor(1.0)
                ),
            ]
        ),
    ]
)
@pytest.mark.local
def test_local(case: MetricCase, device: str):
    assert_metric_local(case, device)


@pytest.mark.parametrize(
    "case",
    [
        # Case: Distributed Homogeneous (All ranks perfect)
        MetricCase(
            factory_fn=BinaryAccuracyMetric,
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep(
                    expect=torch.tensor(1.0),
                    params_per_rank=[
                        MetricParams(torch.tensor([0.9]), torch.tensor([1]))
                        for _ in range(8)
                    ]
                )
            ]
        ),
        # Case: Split Accuracy
        # Ranks 0-3: 100% Accuracy (1 sample each)
        # Ranks 4-7: 0% Accuracy (1 sample each)
        # Global: 4 Correct / 8 Total = 0.5
        MetricCase(
            factory_fn=BinaryAccuracyMetric,
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep(
                    expect=torch.tensor(0.5),
                    params_per_rank=[
                        # Correct predictions
                        MetricParams(torch.tensor([0.9]), torch.tensor([1])),
                        MetricParams(torch.tensor([0.1]), torch.tensor([0])),
                        MetricParams(torch.tensor([0.9]), torch.tensor([1])),
                        MetricParams(torch.tensor([0.1]), torch.tensor([0])),
                        # Incorrect predictions
                        MetricParams(torch.tensor([0.1]), torch.tensor([1])),
                        MetricParams(torch.tensor([0.9]), torch.tensor([0])),
                        MetricParams(torch.tensor([0.1]), torch.tensor([1])),
                        MetricParams(torch.tensor([0.9]), torch.tensor([0])),
                    ]
                )
            ]
        ),
        # Case: Heavy Imbalance
        # Rank 0: 100 items, all correct
        # Rank 1: 100 items, all wrong
        # Ranks 2-7: No data (Empty tensors)
        # Global: 100 / 200 = 0.5
        MetricCase(
            factory_fn=BinaryAccuracyMetric,
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep(
                    expect=torch.tensor(0.5),
                    params_per_rank=[
                        # Rank 0: 100 correct
                        MetricParams(torch.ones(100), torch.ones(100)),
                        # Rank 1: 100 wrong (Pred 1, Label 0)
                        MetricParams(torch.ones(100), torch.zeros(100)),
                        # Others empty
                        MetricParams(torch.tensor([]), torch.tensor([])),
                        MetricParams(torch.tensor([]), torch.tensor([])),
                        MetricParams(torch.tensor([]), torch.tensor([])),
                        MetricParams(torch.tensor([]), torch.tensor([])),
                        MetricParams(torch.tensor([]), torch.tensor([])),
                        MetricParams(torch.tensor([]), torch.tensor([])),
                    ]
                )
            ]
        ),
    ]
)
@pytest.mark.distributed
def test_distributed(dist_ctx_factory, case: MetricCase):
    assert_metric_distributed(dist_ctx_factory, case)
