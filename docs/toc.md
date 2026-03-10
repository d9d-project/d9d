---
icon: lucide/table-of-contents
---

# Table of Contents

## 🌐 Distributed Core
The foundational primitives managing the cluster.

*   **[Distributed Context](./core/dist_context.md)**: The Source of Truth for topology. Understanding `DeviceMesh` domains (`dense`, `expert`, `batch`).
*   **[Distributed Operations](./core/dist_ops.md)**: Utilities for gathering var-length tensors and objects.
*   **[PyTree Sharding](./core/sharding.md)**: Utilities for splitting complex nested structures across ranks.
*   **[Typing Extensions](./core/types.md)**: Python type annotations for common objects and structures.


## 🚀 Execution Engine
How to configure and run jobs.

*   **[Training Loop](./loop/train.md)**: The lifecycle of the `Trainer`, dependency injection, and execution flow.
*   **[Inference Loop](./loop/inference.md)**: The lifecycle of distributed `Inference` and forward-only execution.
*   **[Configuration](./loop/config.md)**: Pydantic schemas for configuring jobs, batching, and logging.
*   **[Interfaces (Providers & Tasks)](./loop/interfaces/index.md)**: How to inject your custom Model, Dataset, and Step logic (Train & Infer).


## 💾 Data & State
Managing data loading and model checkpoints.

*   **[Model State Mapper](./model_states/mapper.md)**: The graph-based transformation engine for checkpoints (transform architectures on-the-fly).
*   **[Model State I/O](./model_states/io.md)**: Streaming reader/writers for checkpoints.
*   **[Datasets](./dataset/index.md)**: Distributed-aware dataset wrappers and smart bucketing.


## 🧠 Modeling & Architecture
Building blocks for modern LLMs.

*   **[Model Catalogue](./models/model_catalogue/index.md)**: Models available directly in d9d.
*   **[Model Design](./models/model_design.md)**: Principles for creating compatible models.
*   **[Modules](./models/modules/index.md)**: Building blocks for implementing compatible models.

## ⚡ Parallelism
Strategies for distributing computations.

*   **[Horizontal Parallelism](./models/horizontal_parallelism.md)**: Data Parallelism, Fully-Sharded Data Parallelism, Expert Parallelism, Tensor Parallelism.
*   **[Pipeline Parallelism](./models/pipeline_parallelism.md)**: Vertical scaling, schedules (1F1B, ZeroBubble), and cross-stage communication.

## 🔧 Fine-Tuning (PEFT)
Parameter-Efficient Fine-Tuning framework.

*   **[Overview](./peft/overview.md)**: Injection lifecycle and state mapping.
*   **Methods**: [LoRA](./peft/lora.md), [Full Tune](./peft/full_tune.md), and [Method Stacking](./peft/stack.md).

## 📈 Optimization & Metrics

*   **[Metrics](./metric/overview.md)**: Distributed-aware statistic accumulation.
*   **[Metric Catalogue](./metric/metric_catalogue/index.md)**: Ready-to-use metric implementations.
*   **[Custom Metrics](./metric/custom.md)**: Implementing custom metrics.
*   **[Experiment Tracking](./internals/tracker_integration.md)**: Integration with logging backends (WandB, Aim).
*   **[Piecewise Scheduler](./lr_scheduler/piecewise.md)**: Composable LR schedules and [Visualization](./lr_scheduler/visualization.md).
*   **[Stochastic Optimizers](./optimizer/stochastic.md)**: Low-precision training using stochastic rounding.

## ⚙️ Internals
Deep dive into the engine room.

*   **[AutoGrad Extensions](./core/autograd_extensions.md)**: How we do split-backward for Pipeline Parallel.
*   **[Pipelining Internals](./internals/pipelining.md)**: How the VM and Schedules work.
*   **[Gradient Sync](./internals/grad_sync.md)**: Custom backward hooks for overlapping comms.
*   **[Gradient Norm & Clipping](./internals/grad_norm.md)**: Correct global norm calculation across hybrid meshes.
*   **[Metric Collection](./internals/metric_collector.md)**: Custom overlapped metric synchronization & computation.
*   **[Pipeline State](./internals/pipeline_state.md)**: Context switching between Global and Microbatch scopes.
*   **[Determinism](./internals/determinism.md)**.
*   **[Profiling](./internals/profiling.md)**.
