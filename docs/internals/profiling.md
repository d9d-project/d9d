# Profiling

## About

The `d9d.internals.profiling` package wraps the PyTorch profiler for distributed jobs.

Profiling a large distributed job has three problems:

1.  **File naming**: Thousands of ranks that write to the same file name cause race conditions.
2.  **Storage space**: Raw Chrome trace JSON files can grow to several GiB quickly.
3.  **Synchronization**: All ranks must profile the same steps without manual work.

The `Profiler` class solves them. It names each trace file after the `DeviceMesh` coordinates of the rank. It compresses each trace into a `.tar.gz` archive right after export. It runs the same periodic schedule (wait, warmup, active) on all ranks.

!!! warning "Internal API"
    If you use the standard d9d training loop, you do not need to call this package. d9d profiles the job based on its configuration. This page is for users who extend d9d.

## API Reference

::: d9d.internals.profiling
