import pytest
from d9d.dist_context import METRIC_DOMAIN, DeviceMeshParameters, MetricDim
from d9d.metric import AsyncMetricCollector

from d9d_dev.metric.collector.metric import MockMetric


@pytest.mark.local
def test_lifecycle_local():
    metric = MockMetric()
    collector = AsyncMetricCollector(metric)

    collector.bind()
    assert str(metric.current_device).startswith("cuda")

    metric.update(10.0)

    collector.schedule_collection(None)

    # Verify sync was NOT called yet (no group), compute WAS called (queued), reset NOT called
    assert not metric.sync_called
    assert metric.compute_called
    assert not metric.reset_called

    result = collector.collect_results()

    assert result.item() == 20.0

    assert metric.reset_called

    collector.unbind()


@pytest.mark.distributed
def test_lifecycle_distributed(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel=8))

    metric = MockMetric()
    collector = AsyncMetricCollector(metric)
    collector.bind()

    metric.update(10.0)

    collector.schedule_collection(dist_ctx.mesh_for(METRIC_DOMAIN)[MetricDim.PP_DP_CP].get_group())

    assert metric.sync_called
    assert metric.compute_called
    assert not metric.reset_called

    result = collector.collect_results()

    # Logic: (10 + 1 (sync add)) * 2 = 22
    assert result.item() == 22.0
    assert metric.reset_called

    collector.unbind()


@pytest.mark.local
def test_state_errors():
    metric = MockMetric()

    collector = AsyncMetricCollector(metric)

    with pytest.raises(RuntimeError, match="not bound"):
        collector.schedule_collection(None)

    with pytest.raises(RuntimeError, match="not bound"):
        collector.collect_results()

    collector.bind()

    with pytest.raises(RuntimeError, match="was not called"):
        collector.collect_results()

    collector.unbind()


@pytest.mark.local
def test_pytree_support():
    class PyTreeMetric(MockMetric):
        def compute(self):
            super().compute()
            return {"a": self.val, "b": [self.val + 1, self.val + 2]}

    metric = PyTreeMetric()
    collector = AsyncMetricCollector(metric)
    collector.bind()

    metric.update(1.0)  # val = 1

    collector.schedule_collection(None)
    results = collector.collect_results()

    # "a" is the metric state itself, which reset() clears, so the results must not alias it.
    assert results["a"].item() == 1.0
    assert [value.item() for value in results["b"]] == [2.0, 3.0]

    collector.unbind()
