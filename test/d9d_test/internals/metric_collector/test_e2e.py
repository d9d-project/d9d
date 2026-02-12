import pytest
from d9d.core.dist_context import DeviceMeshParameters
from d9d.internals.metric_collector.collector import AsyncMetricCollector

from d9d_test.internals.metric_collector.metric import MockMetric


@pytest.mark.local
def test_lifecycle_local(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())
    metric = MockMetric()
    collector = AsyncMetricCollector(metric)

    collector.bind()
    assert str(metric.current_device).startswith("cuda")

    metric.update(10.0)

    collector.schedule_collection(dist_ctx)

    # Verify sync was NOT called yet (not distributed), compute WAS called (queued), reset NOT called
    assert not metric.sync_called
    assert metric.compute_called
    assert not metric.reset_called

    result = collector.collect_results()

    assert isinstance(result, float)
    assert result == 20.0

    assert metric.reset_called

    collector.unbind()


@pytest.mark.distributed
def test_lifecycle_distributed(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters(data_parallel_replicate=8))

    metric = MockMetric()
    collector = AsyncMetricCollector(metric)
    collector.bind()

    metric.update(10.0)

    collector.schedule_collection(dist_ctx)

    assert metric.sync_called
    assert metric.compute_called
    assert not metric.reset_called

    result = collector.collect_results()

    # Logic: (10 + 1 (sync add)) * 2 = 22
    assert result == 22.0
    assert metric.reset_called

    collector.unbind()


@pytest.mark.local
def test_state_errors(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    metric = MockMetric()

    collector = AsyncMetricCollector(metric)

    with pytest.raises(RuntimeError, match="not bound"):
        collector.schedule_collection(dist_ctx)

    with pytest.raises(RuntimeError, match="not bound"):
        collector.collect_results()

    collector.bind()

    with pytest.raises(RuntimeError, match="was not called"):
        collector.collect_results()

    collector.unbind()


@pytest.mark.local
def test_pytree_support(dist_ctx_factory):
    dist_ctx = dist_ctx_factory(DeviceMeshParameters())

    class PyTreeMetric(MockMetric):
        def compute(self):
            super().compute()
            return {"a": self.val, "b": [self.val + 1, self.val + 2]}

    metric = PyTreeMetric()
    collector = AsyncMetricCollector(metric)
    collector.bind()

    metric.update(1.0)  # val = 1

    collector.schedule_collection(dist_ctx)
    results = collector.collect_results()

    assert results["a"] == 1.0
    assert isinstance(results["a"], float)

    assert isinstance(results["b"][0], float)
    assert isinstance(results["b"][1], float)
    assert results["b"] == [2.0, 3.0]

    collector.unbind()
