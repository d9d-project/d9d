import pytest
import torch
from d9d.metric import AsyncMetricCollector

from d9d_dev.metric.collector.metric import MockMetric


@pytest.mark.local
def test_runs_on_side_stream():
    main_stream = torch.cuda.current_stream()
    captured_stream = None

    class StreamCaptureMetric(MockMetric):
        def compute(self):
            nonlocal captured_stream
            captured_stream = torch.cuda.current_stream()
            return super().compute()

    metric = StreamCaptureMetric()
    collector = AsyncMetricCollector(metric)
    collector.bind()

    metric.update(5.0)
    collector.schedule_collection(None)

    assert captured_stream is not None
    assert captured_stream != main_stream
    assert captured_stream == collector._stream

    collector.collect_results()
    collector.unbind()
