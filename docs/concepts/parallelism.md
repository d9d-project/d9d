# Choosing Parallelism

## About

To choose the parallelism strategies of a job and apply them in d9d, use these sources:

*   **What each strategy costs and how to combine them**: The [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) explains data, fully sharded, tensor, context, expert and pipeline parallelism with measurements.
*   **How to apply a strategy in d9d**: [Horizontal Parallelism](../models/horizontal_parallelism.md) covers data, fully sharded and expert parallelism. [Pipeline Parallelism](../models/pipeline_parallelism.md) covers the stages and the schedules.
*   **How to launch a job on several GPUs**: See [Running Jobs](../guides/running.md).
