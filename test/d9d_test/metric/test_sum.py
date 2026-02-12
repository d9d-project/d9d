import pytest
import torch
from d9d.metric.impl import SumMetric

from d9d_test.metric.infra import MetricCase, MetricParams, MetricStep, assert_metric_distributed, assert_metric_local


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "case",
    [
        # Case: SumMetric Basic
        MetricCase(
            factory_fn=SumMetric,
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep(
                    # Simple accumulation
                    [MetricParams(torch.tensor([1.0]))],
                    expect=torch.tensor(1.0),
                ),
                MetricStep(
                    # Accumulate a vector (should sum components)
                    [MetricParams(torch.tensor([2.0, 3.0]))],
                    expect=torch.tensor(6.0),  # 1 + 2 + 3
                ),
                MetricStep(
                    # Accumulate a matrix
                    [MetricParams(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))],
                    expect=torch.tensor(12.0),  # 6 + (1+1+2+2)
                ),
            ],
        ),
        # Case: SumMetric Empty
        MetricCase(
            factory_fn=SumMetric,
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep([MetricParams(torch.tensor([]))], expect=torch.tensor(0.0)),
                MetricStep([MetricParams(torch.tensor([10.5]))], expect=torch.tensor(10.5)),
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
        # Case: SumMetric Simple Distributed Sum
        # Each rank adds 10. Total 8 ranks. Expected sum = 80.
        MetricCase(
            factory_fn=SumMetric,
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep(
                    expect=torch.tensor(80.0), params_per_rank=[MetricParams(torch.tensor(10.0)) for _ in range(8)]
                ),
                MetricStep(
                    expect=torch.tensor(160.0), params_per_rank=[MetricParams(torch.tensor(10.0)) for _ in range(8)]
                ),
            ],
        ),
        # Case: SumMetric Rank-dependent
        # Rank i adds i. Sum 0..7 = 28.
        MetricCase(
            factory_fn=SumMetric,
            initial_expect=torch.tensor(0.0),
            steps=[
                MetricStep(
                    expect=torch.tensor(28.0), params_per_rank=[MetricParams(torch.tensor(float(i))) for i in range(8)]
                )
            ],
        ),
    ],
)
@pytest.mark.distributed
def test_distributed(dist_ctx_factory, case: MetricCase):
    assert_metric_distributed(dist_ctx_factory, case)
