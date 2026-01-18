---
title: Introduction
---

# Introduction to Pipelining

## About Pipeline Parallelism

Pipeline Parallelism is a technique to train large models that do not fit into the GPU memory of a single device. Unlike Tensor Parallelism (which splits individual tensors across devices) or Fully-Sharded Data Parallelism that splits the model horizontally, Pipeline Parallelism splits the model **vertically** by layers.

Imagine a model with 32 layers running on 4 GPUs.

*   **GPU 0** holds layers 0-7.
*   **GPU 1** holds layers 8-15.
*   ...and so on.

Data flows through these GPUs sequentially. To prevent GPUs from idling while waiting for previous chunks (the "bubble" problem), the input batch is split into smaller **micro-batches** which are pipelined through the stages.

For detailed information about pipelining, please see the [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook).
