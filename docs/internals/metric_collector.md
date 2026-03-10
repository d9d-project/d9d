# Async Metric Collection

!!! warning "Internal API Warning"
    If you are using the standard `d9d` `Trainer`, you **do not** need to interact with this package directly. It is handled automatically. This documentation is intended for users implementing custom training loops or logging infrastructure.

## About

The `d9d.internals.metric_collector` package provides the infrastructure for non-blocking metric processing. 

While the [`Metric`](../metric/overview.md) interface is synchronous by design, the `AsyncMetricCollector` wraps a metric instance and schedules its synchronization and computation on a secondary CUDA stream. This allows the main training loop to proceed immediately without waiting for metric reductions (all-reduce) to complete.

::: d9d.internals.metric_collector
