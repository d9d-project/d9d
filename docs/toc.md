---
icon: lucide/table-of-contents
---

# Table of Contents

## 🏁 Getting Started
Your first job.

*   **[Installation](./getting_started/installation.md)**: Install d9d from PyPI or from a checkout, and the optional extras.

## 🌐 Distributed Core
The foundational primitives managing the cluster.

*   **[Distributed Context](./core/dist_context.md)**: The source of truth for topology, and the `DeviceMesh` domains (`regular`, `dense`, `expert`, `batch`, `flat`).
*   **[Distributed Operations](./core/dist_ops.md)**: Utilities for gathering var-length tensors and objects.
*   **[State Offloading](./core/offload.md)**: Releasing GPU training state to host memory for colocated RL (the sleep/wake primitives).
*   **[PyTree Traversal](./core/pytree.md)**: Dataclass-aware recursive mapping and flattening over nested tensor structures.
*   **[Typing Extensions](./core/types.md)**: Python type annotations for common objects and structures.


## 🚀 Execution Engine
How to configure and run jobs.

*   **[Training Loop](./loop/train.md)**: The lifecycle of the `Trainer`, dependency injection, and execution flow.
*   **[Inference Loop](./loop/inference.md)**: The lifecycle of distributed `Inference` and forward-only execution.
*   **[Configuration](./loop/config.md)**: Pydantic schemas for configuring jobs, scheduling, and logging.
*   **[Interfaces](./loop/interfaces/index.md)**: How to inject your custom Model, Data, and Step logic (Train & Infer).


## 💾 Data and State
Managing data loading and model checkpoints.

*   **[Model State Mapper](./model_states/mapper.md)**: The graph-based transformation engine for checkpoints (transform architectures on-the-fly).
*   **[Model State I/O](./model_states/io.md)**: Streaming reader/writers for checkpoints.
*   **[Datasets](./dataset/index.md)**: Distributed-aware dataset wrappers and length bucketing.


## 🧠 Modeling and Architecture
Building blocks for LLMs.

*   **[Model Catalogue](./models/model_catalogue/index.md)**: Models available directly in d9d.
*   **[Model Design](./models/model_design.md)**: Principles for creating compatible models.
*   **[Modules](./models/modules/index.md)**: Building blocks for implementing compatible models.

## ⚡ Parallelism
Strategies for distributing computations.

*   **[Horizontal Parallelism](./models/horizontal_parallelism.md)**: Data parallelism, Fully Sharded Data Parallel (FSDP), expert parallelism and tensor parallelism.
*   **[Pipeline Parallelism](./models/pipeline_parallelism.md)**: Vertical scaling, schedules (1F1B, Zero Bubble), and cross-stage communication.

## 🔧 Fine-Tuning (PEFT)
Parameter-Efficient Fine-Tuning framework.

*   **[PEFT Overview](./peft/overview.md)**: Injection lifecycle and state mapping.
*   **Methods**: [LoRA](./peft/lora.md), [Full Fine-Tuning](./peft/full_tune.md), and [Method Stacking](./peft/stack.md).

## 📈 Optimization and Metrics
Metrics, learning rate schedules and optimizers.

*   **[Metrics Overview](./metric/overview.md)**: Distributed-aware statistic accumulation.
*   **[Metric Catalogue](./metric/metric_catalogue/index.md)**: Ready-to-use metric implementations.
*   **[Custom Metrics](./metric/custom.md)**: Implementing custom metrics.
*   **[Piecewise Scheduler](./lr_scheduler/piecewise.md)**: Composable LR schedules and [Schedule Visualization](./lr_scheduler/visualization.md).
*   **[Stochastic Optimizers](./optimizer/stochastic.md)**: Low-precision training using stochastic rounding.

## 📊 Experiment Tracking
Where the loss and the metrics go.

*   **[Experiment Tracking](./tracker/index.md)**: Choosing a tracker (the log, Aim), what the loop logs and adding a new tracker.

## ⚙️ Internals
Deep dive into the engine room.

*   **[Autograd Extensions](./core/autograd_extensions.md)**: How we do split backward for pipeline parallelism.
*   **[Pipelining Internals](./internals/pipelining.md)**: How the VM and Schedules work.
*   **[Gradient Synchronization](./internals/grad_sync.md)**: Custom backward hooks for overlapping comms.
*   **[Gradient Norm and Clipping](./internals/grad_norm.md)**: Correct global norm calculation across hybrid meshes.
*   **[Metric Collection](./internals/metric_collector.md)**: Custom overlapped metric synchronization & computation.
*   **[Determinism](./internals/determinism.md)**: RNG seeding across distributed processes.
*   **[Profiling](./internals/profiling.md)**: A distributed-aware wrapper around the PyTorch Profiler.
