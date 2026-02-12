import pytest
import torch
from d9d.metric.impl import WeightedMeanMetric

from d9d_test.metric.infra import MetricCase, MetricParams, MetricStep, assert_metric_distributed, assert_metric_local


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "case",
    [
        # Case: Default WM
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    [MetricParams(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))], expect=torch.tensor(1.5)
                ),
                MetricStep([MetricParams(torch.tensor(4.0), torch.tensor(0.0))], expect=torch.tensor(1.5)),
            ],
        ),
        # Case: Zero weights (Masking/Padding scenario) WM
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    # Initial state: 10/1 = 10
                    [MetricParams(torch.tensor([10.0]), torch.tensor([1.0]))],
                    expect=torch.tensor(10.0),
                ),
                MetricStep(
                    # Update with weight 0. Should ignore value 500 completely.
                    # State: 10/1 = 10
                    [MetricParams(torch.tensor([500.0]), torch.tensor([0.0]))],
                    expect=torch.tensor(10.0),
                ),
                MetricStep(
                    # Valid update.
                    # State: (10 + 20) / (1 + 1) = 15
                    [MetricParams(torch.tensor([20.0]), torch.tensor([1.0]))],
                    expect=torch.tensor(15.0),
                ),
            ],
        ),
        # Case: Multi-dimensional tensors (Batch handling) WM
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    # 2x2 Batch. Sum=10, WeightSum=4 -> 2.5
                    [MetricParams(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([[1.0, 1.0], [1.0, 1.0]]))],
                    expect=torch.tensor(2.5),
                ),
                MetricStep(
                    # Add heavy weighted items.
                    # Previous Num: 10, Denom: 4.
                    # Update Num: (10*8) + (10*8) = 160. Update Denom: 8+8 = 16.
                    # Total Num: 170. Total Denom: 20. Result: 8.5
                    [MetricParams(torch.tensor([10.0, 10.0]), torch.tensor([8.0, 8.0]))],
                    expect=torch.tensor(8.5),
                ),
            ],
        ),
        # Case: Empty/Zero-sized tensors (Edge case logic) WM
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    # Update with empty tensors -> No change (remains NaN)
                    [MetricParams(torch.tensor([]), torch.tensor([]))],
                    expect=torch.tensor(torch.nan),
                ),
                MetricStep(
                    # First valid update
                    [MetricParams(torch.tensor([5.0]), torch.tensor([1.0]))],
                    expect=torch.tensor(5.0),
                ),
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
        # Case: Default WM (Homogeneous updates then zero-weight updates)
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    expect=torch.tensor(1.5),
                    params_per_rank=[
                        MetricParams(torch.tensor(1.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(1.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(1.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(1.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(2.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(2.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(2.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(2.0), torch.tensor(1.0)),
                    ],
                ),
                MetricStep(
                    expect=torch.tensor(1.5),
                    params_per_rank=[
                        MetricParams(torch.tensor(11342.0), torch.tensor(0.0)),
                        MetricParams(torch.tensor(1231.0), torch.tensor(0.0)),
                        MetricParams(torch.tensor(123123123.0), torch.tensor(0.0)),
                        MetricParams(torch.tensor(12.0), torch.tensor(0.0)),
                        MetricParams(torch.tensor(12.0), torch.tensor(0.0)),
                        MetricParams(torch.tensor(121.0), torch.tensor(0.0)),
                        MetricParams(torch.tensor(1212.0), torch.tensor(0.0)),
                        MetricParams(torch.tensor(2121.0), torch.tensor(0.0)),
                    ],
                ),
            ],
        ),
        # Case: Ranks with uneven data (simulating data imbalance or masking)
        # Ranks 0-3 provide Value 10, Weight 1.
        # Ranks 4-7 provide Value 100, Weight 0 (should be ignored).
        # Expected Mean: 10.0
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    expect=torch.tensor(10.0),
                    params_per_rank=[
                        # Active Ranks
                        MetricParams(torch.tensor(10.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(10.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(10.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(10.0), torch.tensor(1.0)),
                        # Inactive Ranks (weights=0)
                        MetricParams(torch.tensor(100.0), torch.tensor(0.0)),
                        MetricParams(torch.tensor(100.0), torch.tensor(0.0)),
                        MetricParams(torch.tensor(100.0), torch.tensor(0.0)),
                        MetricParams(torch.tensor(100.0), torch.tensor(0.0)),
                    ],
                )
            ],
        ),
        # Case: Step-wise accumulation
        # Step 1: All ranks contribute 0. Mean 0.
        # Step 2: All ranks contribute 20. Global state: sum_val=(0*8 + 20*8)=160, sum_w=(8+8)=16. Mean=10.
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    expect=torch.tensor(0.0),
                    params_per_rank=[MetricParams(torch.tensor(0.0), torch.tensor(1.0)) for _ in range(8)],
                ),
                MetricStep(
                    expect=torch.tensor(10.0),
                    params_per_rank=[MetricParams(torch.tensor(20.0), torch.tensor(1.0)) for _ in range(8)],
                ),
            ],
        ),
        # Case: Rank-dependent values (Verification of actual reduction)
        # Rank i contributes Value=i, Weight=1.
        # Sum = 0+1+2+3+4+5+6+7 = 28. Total Weight = 8. Mean = 3.5.
        MetricCase(
            factory_fn=WeightedMeanMetric,
            initial_expect=torch.tensor(torch.nan),
            steps=[
                MetricStep(
                    expect=torch.tensor(3.5),
                    params_per_rank=[
                        MetricParams(torch.tensor(0.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(1.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(2.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(3.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(4.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(5.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(6.0), torch.tensor(1.0)),
                        MetricParams(torch.tensor(7.0), torch.tensor(1.0)),
                    ],
                )
            ],
        ),
    ],
)
@pytest.mark.distributed
def test_distributed(dist_ctx_factory, case: MetricCase):
    assert_metric_distributed(dist_ctx_factory, case)
