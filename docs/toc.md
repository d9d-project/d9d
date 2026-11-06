---
icon: lucide/table-of-contents
---

# Table of Contents

## 🏁 Getting Started
Your first job.

*   **[Installation](./getting_started/installation.md)**: Install d9d from PyPI or from a checkout, and the optional extras.
*   **[Quickstart](./getting_started/quickstart.md)**: Train a small model on one GPU and learn the parts of a d9d job.

## 💡 Concepts
The ideas behind d9d.

*   **[How d9d Works](./concepts/how_d9d_works.md)**: The parts of a job, the model build, parameter placements, packs and pipeline programs, the training loop, checkpoints and events.
*   **[Choosing Parallelism](./concepts/parallelism.md)**: Where to learn the parallelism strategies and how to apply them in d9d.
*   **[d9d and Other Frameworks](./concepts/comparison.md)**: How the design of d9d compares with Megatron-Core, torchtitan, DeepSpeed, Accelerate and others, and when to choose each.

## 🍳 Guides
Solutions for concrete tasks.

*   **[Write Your Own Job](./guides/write_your_own_job.md)**: Turn the Quickstart script into a job for your own model, data and loss.
*   **[Running Jobs](./guides/running.md)**: Launch on one GPU, several GPUs or several nodes. Checkpoints, resuming and export.
*   **[Fine-Tune a Hugging Face Model](./guides/finetune_huggingface.md)**: Load Qwen3 from a Hugging Face checkpoint, fine-tune all weights or LoRA adapters and export it back.
*   **[Troubleshooting](./guides/troubleshooting.md)**: Reading the logs of a multi-rank job.

## 🚀 Execution Engine
How to configure and run jobs.

*   **[Training Loop](./loop/train.md)**: How the `Trainer` builds a job and runs its steps, data prefetching, profiling, and sleep and wake.
*   **[Inference Loop](./loop/inference.md)**: How the `Inference` engine runs forward passes over a dataset.
*   **[Checkpointing](./loop/checkpointing.md)**: What a checkpoint holds, when the job saves, how it resumes and where it must be stored.
*   **[Random Numbers in the Loop](./loop/determinism.md)**: How the loop seeds each call of user code, and the rules for random numbers in your code.
*   **[Configuration](./loop/config.md)**: The Pydantic models of the training and inference configuration.
*   **[Interfaces](./loop/interfaces/index.md)**: The providers and tasks that plug your model, data, loss and optimizer into the loop.

## 💾 Data
Datasets and the streams of microbatches that a job consumes.

*   **[Data](./data/index.md)**: Dataset wrappers for bucketing, sharding and padding, and the streams of microbatch packs.

## 🧠 Modeling
Models, their building blocks and parameter-efficient fine-tuning.

*   **[Model Catalogue](./models/index.md)**: The models that d9d ships, with their parallelism and Hugging Face state mappers.
*   **[Decoder with Heads](./models/decoder.md)**: A decoder backbone composed with one or several task heads, and the pipeline I/O types.
*   **[Model Design](./models/model_design.md)**: The rules that your own model must follow to run in the d9d loop.
*   **[Modules](./nn/index.md)**: The building blocks of models: attention, MoE, normalization, heads and other layers.
*   **[PEFT Overview](./peft/overview.md)**: How a PEFT method changes the model, which parameters it trains and how it maps the checkpoint.
*   **Methods**: [LoRA](./peft/lora.md), [Full Fine-Tuning](./peft/full_tune.md), and [Method Stacking](./peft/stack.md).

## 📦 Model State
Loading and exporting model weights, including Hugging Face checkpoints.

*   **[Model State Mapper](./model_state/mapper.md)**: Graphs of mappers that transform checkpoint keys and tensors into the model layout and back.
*   **[Model State I/O](./model_state/io.md)**: Reading and writing checkpoints in a stream, with a mapper applied on the way.

## ⚡ Parallelism
Strategies for distributing computations.

*   **[Distributed Context](./dist_context/index.md)**: The `DistributedContext` and the `DeviceMesh` domains (`regular`, `dense`, `expert`, `batch`, `flat`).
*   **[Horizontal Parallelism](./parallel/index.md)**: Replicate, FSDP, HSDP, expert parallelism and custom strategies over `DTensor` placements.
*   **[Parameter Runtime](./parallel/runtime.md)**: DDP, FSDP and HSDP run from the pipeline program, without hooks.
*   **[Parameter Runtime Internals](./parallel/runtime_internals.md)**: Buckets, collectives, the memory budget and the reductions of the pipeline program.
*   **[Gradient Norm and Clipping](./parallel/grad_norm.md)**: The global gradient norm and clipping over parameters with mixed placements.
*   **[Pipeline Parallelism](./pipelining/index.md)**: Pipeline stages, the schedules from GPipe to DualPipeV and how a model supports them.
*   **[Pipelining Internals](./pipelining/internals.md)**: How the pipeline virtual machine runs programs, and how to write a new schedule.
*   **[Split Backward](./autograd/split_backward.md)**: Separate backward passes for input and weight gradients, and the context that custom autograd functions check.

## 📉 Optimization
Optimizers and learning rate schedules.

*   **[Stochastic Optimizers](./optim/stochastic.md)**: Optimizers that keep bf16 parameters without an fp32 copy, with stochastic rounding.
*   **[Piecewise Scheduler](./optim/lr_scheduler/piecewise.md)**: Learning rate schedules built from phases, and how to [plot them](./optim/lr_scheduler/visualization.md).

## 📊 Metrics and Tracking
What the job measures and where it goes.

*   **[Metrics Overview](./metric/overview.md)**: The `Metric` interface and its lifecycle: update, sync, compute and reset.
*   **[Metric Catalogue](./metric/catalogue/index.md)**: The metrics that d9d ships, e.g. a weighted mean, AUROC and the metrics of a confusion matrix.
*   **[Custom Metrics](./metric/custom.md)**: How to write your own metric with `MetricAccumulator`.
*   **[Metric Collection](./metric/collector.md)**: Synchronizing and computing a metric on a side CUDA stream, without blocking the loop.
*   **[Experiment Tracking](./tracker/index.md)**: The trackers (the log, W&B, TensorBoard, MLflow, Aim), what the loop logs and how to add a tracker.

## 🧰 Utilities
Low-level helpers: nested tensor structures, moving GPU state to host memory and seeds.

*   **[PyTree Traversal](./pytree/index.md)**: Mapping and flattening nested tensor structures, including dataclasses.
*   **[State Offloading](./offload/index.md)**: Moving GPU training state to host memory and back, the base of sleep and wake.
*   **[Random Number Generation](./rng/index.md)**: Deriving seeds from a base seed and keys, equal in every process, and forking the global generators with them.
