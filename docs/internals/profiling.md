---
title: Distributed Profiling
---

# Distributed Profiling

## About

The `d9d.internals.profiling` package provides a distributed-aware wrapper around the standard PyTorch Profiler. 

In large-scale distributed training, profiling often becomes difficult due to:

1.  **File Naming**: Thousands of ranks writing to the same filename causes race conditions.
2.  **Storage Space**: Raw Chrome tracing JSON files can grow to gigabytes very quickly.
3.  **Synchronization**: Ensuring all ranks profile the same specific step without manual intervention.

The `Profiler` class solves these issues by automatically handling file naming based on the `DeviceMesh` coordinates, compressing traces into `.tar.gz` archives on the fly, and managing the profiling schedule (wait/warmup/active).

::: d9d.internals.profiling
    options:
        show_root_heading: true
        show_root_full_path: true
