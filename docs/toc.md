---
title: Table of Contents
---

## 🌐 Distributed Core
The foundational primitives managing the cluster.

*   **[Distributed Context](./core/dist_context.md)**: The Source of Truth for topology. Understanding `DeviceMesh` domains (`dense`, `expert`, `batch`).


## 🚀 Execution Engine
How to configure and run jobs.

*   **[Training Loop](./0_loop/0_index.md)**: The lifecycle of the `Trainer`, dependency injection, and execution flow.
*   **[Inference Loop](./0_loop/infer.md)**: The lifecycle of distributed `Inference` and forward-only execution.
*   **[Configuration](./0_loop/config.md)**: Pydantic schemas for configuring jobs, batching, and logging.
*   **[Interfaces (Providers & Tasks)](./0_loop/interfaces.md)**: How to inject your custom Model, Dataset, and Step logic (Train & Infer).


## 💾 Data & State
Managing data loading and model checkpoints.

*   **[Model State Mapper](./model_states/mapper.md)**: The graph-based transformation engine for checkpoints (transform architectures on-the-fly).
*   **[Model State I/O](./model_states/io.md)**: Streaming reader/writers for checkpoints.
*   **[Datasets](./dataset/index.md)**: Distributed-aware dataset wrappers and smart bucketing.


## 🧠 Modeling & Architecture
Building blocks for modern LLMs.

*   **[Model Design](./models/1_model_design.md)**: Principles for creating compatible models.
*   **[Model Catalogue](./models/4_model_catalogue.md)**: Models available directly in d9d.
*   **Building Blocks**:
    *   [Attention](./modules/attention.md) (FlashSDPA, GQA)
    *   [Mixture of Experts](./modules/moe.md) (Sparse Experts, Routers, DeepEP integration)
    *   [Heads](./modules/head.md) & [Embeddings](./modules/embedding.md) (Split Vocab support)
    *   [Positional Embeddings](./modules/positional.md) (RoPE)
    *   [FFN](./modules/ffn.md) (SwiGLU)

## ⚡ Parallelism
Strategies for distributing computations.

*   **[Horizontal Parallelism](./models/2_horizontal_parallelism.md)**: Data Parallelism, Fully-Sharded Data Parallelism, Expert Parallelism, Tensor Parallelism.
*   **[Pipeline Parallelism](./models/3_pipeline_parallelism.md)**: Vertical scaling, schedules (1F1B, ZeroBubble), and cross-stage communication.
    **[Distributed Operations](./core/dist_ops.md)**: Utilities for gathering var-length tensors and objects.
*   **[PyTree Sharding](./core/sharding.md)**: Splitting complex nested structures across ranks.

## 🔧 Fine-Tuning (PEFT)
Parameter-Efficient Fine-Tuning framework.

*   **[Overview](./peft/0_index.md)**: Injection lifecycle and state mapping.
*   **Methods**: [LoRA](./peft/lora.md), [Full Tune](./peft/full_tune.md), and [Method Stacking](./peft/stack.md).

## 📈 Optimization & Metrics

*   **[Metrics](./metric/index.md)**: Distributed-aware statistic accumulation.
*   **[Experiment Tracking](./internals/tracker_integration.md)**: Integration with logging backends (WandB, Aim).
*   **[Piecewise Scheduler](./lr_scheduler/piecewise.md)**: Composable LR schedules and [Visualization](./lr_scheduler/visualization.md).
*   **[Stochastic Optimizers](./optimizer/stochastic.md)**: Low-precision training using stochastic rounding.

## ⚙️ Internals
Deep dive into the engine room.

*   **[AutoGrad Extensions](./core/autograd_extensions.md)**: How we do split-backward for Pipeline Parallel.
*   **[Pipelining Internals](./internals/pipelining.md)**: How the VM and Schedules work.
*   **[Gradient Sync](./internals/grad_sync.md)**: Custom backward hooks for overlapping comms.
*   **[Gradient Norm & Clipping](./internals/grad_norm.md)**: Correct global norm calculation across hybrid meshes.
*   **[Pipeline State](./internals/pipeline_state.md)**: Context switching between Global and Microbatch scopes.
*   **[Determinism](./internals/determinism.md)**.
*   **[Profiling](./internals/profiling.md)**.
