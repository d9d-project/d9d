import numpy as np
import pytest
import torch
from d9d.metric.impl import BinaryAUROCMetric
from sklearn.metrics import roc_auc_score

from d9d_test.metric.infra import MetricCase, MetricParams, MetricStep, assert_metric_distributed, assert_metric_local


def _get_sklearn_truth(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    y_true = labels.flatten().cpu().numpy()
    y_score = preds.flatten().cpu().numpy()

    # Check if we have both classes
    if len(np.unique(y_true)) < 2:
        return torch.tensor(0.5, dtype=torch.float32)

    score = roc_auc_score(y_true, y_score)
    return torch.tensor(score, dtype=torch.float32)


def _new_random_case(seed: int, size: int, num_bins: int = 10000):
    torch.manual_seed(seed)
    preds = torch.rand(size)

    # Correlated noise
    labels = (preds + (torch.rand(size) * 0.8)).round().clamp(0, 1).long()
    expect_rnd = _get_sklearn_truth(preds, labels)

    return MetricCase(
        factory_fn=lambda: BinaryAUROCMetric(num_bins=num_bins),
        initial_expect=torch.tensor(0.5),
        steps=[
            MetricStep(
                [MetricParams(preds, labels)],
                expect=expect_rnd
            )
        ]
    )


def _new_random_distributed_case(seed: int, size: int, num_bins: int = 10000, num_ranks: int = 8):
    torch.manual_seed(seed)
    preds = torch.rand(size)
    labels = (preds + (torch.rand(size) * 0.8)).round().clamp(0, 1).long()

    # Calculate Global Expectation
    expect_global = _get_sklearn_truth(preds, labels)

    # Split data for ranks
    chunk_size = size // num_ranks
    params_per_rank = []

    for i in range(num_ranks):
        start = i * chunk_size
        end = start + chunk_size if i < num_ranks - 1 else size

        p_slice = preds[start:end]
        l_slice = labels[start:end]
        params_per_rank.append(MetricParams(p_slice, l_slice))

    return MetricCase(
        factory_fn=lambda: BinaryAUROCMetric(num_bins=num_bins),
        initial_expect=torch.tensor(0.5),
        steps=[
            MetricStep(
                params_per_rank=params_per_rank,
                expect=expect_global
            )
        ]
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "case",
    [
        # Edge Case - Start with 0, then 1
        MetricCase(
            factory_fn=BinaryAUROCMetric,
            initial_expect=torch.tensor(0.5),
            steps=[
                # Step 1: Only Negatives
                MetricStep(
                    [MetricParams(torch.tensor([0.1, 0.2]), torch.tensor([0, 0]))],
                    expect=torch.tensor(0.5)
                ),
                # Step 2: Add Positives (Now we have valid AUROC)
                # Preds: [0.1, 0.2, 0.8, 0.9], Labels: [0, 0, 1, 1] -> Perfect AUC
                MetricStep(
                    [MetricParams(torch.tensor([0.8, 0.9]), torch.tensor([1, 1]))],
                    expect=torch.tensor(1.0)
                )
            ]
        ),
        # Edge Case - Start with 1, then 0
        MetricCase(
            factory_fn=BinaryAUROCMetric,
            initial_expect=torch.tensor(0.5),
            steps=[
                # Step 1: Only Positives
                MetricStep(
                    [MetricParams(torch.tensor([0.1, 0.2]), torch.tensor([1, 1]))],
                    expect=torch.tensor(0.5)
                ),
                # Step 2: Add Negatives (Now we have valid AUROC)
                # Preds: [0.1, 0.2, 0.8, 0.9], Labels: [0, 0, 1, 1] -> Perfect AUC
                MetricStep(
                    [MetricParams(torch.tensor([0.8, 0.9]), torch.tensor([0, 0]))],
                    expect=torch.tensor(0.0)
                )
            ]
        ),
        # Edge Case - All 0
        MetricCase(
            factory_fn=BinaryAUROCMetric,
            initial_expect=torch.tensor(0.5),
            steps=[
                MetricStep(
                    [MetricParams(torch.tensor([0.1, 0.2]), torch.tensor([0, 0]))],
                    expect=torch.tensor(0.5)
                ),
                MetricStep(
                    [MetricParams(torch.tensor([0.8, 0.9]), torch.tensor([0, 0]))],
                    expect=torch.tensor(0.5)
                )
            ]
        ),
        # Edge Case - All 1
        MetricCase(
            factory_fn=BinaryAUROCMetric,
            initial_expect=torch.tensor(0.5),
            steps=[
                MetricStep(
                    [MetricParams(torch.tensor([0.1, 0.2]), torch.tensor([1, 1]))],
                    expect=torch.tensor(0.5)
                ),
                MetricStep(
                    [MetricParams(torch.tensor([0.8, 0.9]), torch.tensor([1, 1]))],
                    expect=torch.tensor(0.5)
                )
            ]
        ),
    ]
)
@pytest.mark.local
def test_local(case: MetricCase, device: str):
    assert_metric_local(case, device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "case",
    [
        _new_random_case(seed=42, size=10 ** i)
        for i in range(5)
    ]
)
@pytest.mark.local
def test_local_random(case: MetricCase, device: str):
    assert_metric_local(case, device, atol=1e-4, rtol=1.3e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "case",
    [
        _new_random_case(seed=42, size=10 ** i, num_bins=100000)
        for i in range(5)
    ]
)
@pytest.mark.local
def test_local_random_precise(case: MetricCase, device: str):
    assert_metric_local(case, device, atol=1e-5, rtol=1.3e-6)


@pytest.mark.parametrize(
    "case",
    [
        _new_random_distributed_case(seed=42, size=100),
        _new_random_distributed_case(seed=43, size=1000),
    ]
)
@pytest.mark.distributed
def test_distributed_random(dist_ctx_factory, case: MetricCase):
    assert_metric_distributed(dist_ctx_factory, case, atol=1e-4, rtol=1.3e-6)


@pytest.mark.parametrize(
    "case",
    [
        _new_random_distributed_case(seed=42, size=2000, num_bins=100000),
    ]
)
@pytest.mark.distributed
def test_distributed_random_precise(dist_ctx_factory, case: MetricCase):
    assert_metric_distributed(dist_ctx_factory, case, atol=1e-5, rtol=1.3e-6)
