# Metric Collection

## About

The `d9d.internals.metric_collector` package processes metrics without blocking the training loop. The [`Metric`](../metric/overview.md) interface is synchronous. `AsyncMetricCollector` wraps a metric and runs its synchronization and computation on a side CUDA stream. So the training loop continues at once and does not wait for the metric all-reduce. Collecting the results makes the current stream wait for the side stream, but not the host: the results stay tensors on the GPU.

!!! warning "Internal API"
    If you use the standard d9d `Trainer`, you do not need to use this package directly. This page is for users who write their own training loop or logging.

## API Reference

::: d9d.internals.metric_collector
