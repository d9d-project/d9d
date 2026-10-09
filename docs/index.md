---
icon: lucide/house
---

# The d9d Project

**d9d** is a composable framework for distributed training in PyTorch. You do not fit your model into a framework's template: you write it in plain PyTorch and compose the job from d9d parts. d9d runs the job on one GPU or on a multi-node cluster. It combines data, fully sharded, tensor, context, expert and pipeline parallelism in one N-dimensional device mesh.

d9d is for researchers and engineers who change the model, the parallelism or the training method itself. Each change stays in their own code.

!!! warning "Alpha"
    d9d is in alpha, so public APIs can change between minor releases. Read the [changelog](https://github.com/d9d-project/d9d/blob/main/CHANGELOG.md) before you upgrade.

## Get Started

=== "pip"

    ```bash
    pip install d9d
    ```

=== "poetry"

    ```bash
    poetry add d9d
    ```

=== "uv"

    ```bash
    uv add d9d
    ```

*   **[Quickstart](./getting_started/quickstart.md)**: Train a small model on one GPU and learn the parts of a job.
*   **[How d9d Works](./concepts/how_d9d_works.md)**: How the parts of a job fit together.
*   **[Fine-Tune a Hugging Face Model](./guides/finetune_huggingface.md)**: A complete job that loads Qwen3, trains it and exports it back.
*   **[Table of Contents](./toc.md)**: Every page, with a short description.

Some features need extras, such as `d9d[backend-sdpa-flash-attention-4]`. See [Installation](./getting_started/installation.md).

## What You Can Train

d9d runs the same job code from a quick experiment on one consumer GPU, such as an RTX 3090, to a large run on a multi-node cluster. You can use it for:

*   **LLM pre-training**: Dense and MoE language models across many nodes. See [Choosing Parallelism](./concepts/parallelism.md).
*   **Fine-tuning and post-training**: Load a Hugging Face checkpoint, train all weights or LoRA adapters with your own loss, and export the model back. See [Fine-Tune a Hugging Face Model](./guides/finetune_huggingface.md).
*   **Reinforcement learning**: Hand the GPUs to a rollout engine between training steps and take them back. See [Colocated RL](./loop/train.md#sleep-and-wake-colocated-rl).
*   **Classifiers, embedding models and multi-head models**: Put classification, embedding or several heads on a language model backbone, and track metrics such as AUROC and F1 across ranks. See [Model Heads](./models/modules/head.md).
*   **New architectures**: A model that no framework ships yet: a new attention variant, a recurrent or state-space model, or something that is not a transformer at all. Write it in plain PyTorch, reuse d9d blocks where they fit, and train it like any other d9d model. See [Model Design](./models/model_design.md).
*   **New parallelism strategies**: A way to split a model that no framework offers yet, e.g. a new expert placement or a custom tensor-parallel layout for your own layer. Write it as a function that turns the parameters of a submodule into `DTensor`s with your placements. Gradient synchronization and checkpointing read the placements, so the rest of d9d works with it unchanged. See [Horizontal Parallelism](./models/horizontal_parallelism.md).
*   **Evaluation and batch inference**: Run a model over a dataset with the same data, model and parallelism code. See [Inference Loop](./loop/inference.md).
*   **Anything else that trains with a loss**: The model, the data and the loss are your code, e.g. the regression model of the [Quickstart](./getting_started/quickstart.md).

## What You Write

You write four parts: the data, the model, the task with the loss, and the model provider that builds and distributes the model. The data and the model are plain PyTorch, and the task and the model provider implement small d9d interfaces. d9d provides the rest, and you can replace any part. This is the [Quickstart](./getting_started/quickstart.md) job, without its type parameters:

=== "Data"

    ```python
    @dataclass
    class RegressionSample:
        x: torch.Tensor
        y: torch.Tensor


    @dataclass
    class RegressionBatch:
        x: torch.Tensor
        y: torch.Tensor


    # A standard PyTorch dataset. d9d shards it across the data-parallel ranks.
    class RegressionDataset(Dataset):
        def __init__(self, num_samples: int, num_features: int):
            generator = torch.Generator().manual_seed(0)
            self._x = torch.randn(num_samples, num_features, generator=generator)
            true_weight = torch.randn(num_features, 1, generator=generator)
            self._y = self._x @ true_weight + 0.1 * torch.randn(num_samples, 1, generator=generator)

        def __getitem__(self, index: int) -> RegressionSample:
            return RegressionSample(x=self._x[index], y=self._y[index])

        def __len__(self) -> int:
            return len(self._x)

        # Stacks the samples into one microbatch.
        @staticmethod
        def collate(samples: Sequence[RegressionSample]) -> RegressionBatch:
            return RegressionBatch(x=torch.stack([s.x for s in samples]), y=torch.stack([s.y for s in samples]))
    ```

=== "Model"

    ```python
    class MLP(nn.Module):
        def __init__(self, num_features: int, hidden_size: int):
            super().__init__()
            self.up = nn.Linear(num_features, hidden_size)
            self.down = nn.Linear(hidden_size, 1)

        # `inputs` holds what the task passes to the model. This job passes no `shared` data.
        def forward(self, inputs: RegressionInput, shared: None) -> torch.Tensor:
            return self.down(torch.relu(self.up(inputs.x)))

        # d9d builds the model on the meta device and initializes it on the GPU.
        def reset_parameters(self):
            self.up.reset_parameters()
            self.down.reset_parameters()
    ```

=== "Task"

    ```python
    @dataclass
    class RegressionInput:
        x: torch.Tensor


    @dataclass
    class RegressionState:
        y: torch.Tensor


    class RegressionTask(TrainTask):
        def build_forward_inputs(self, ctx: BuildForwardInputsContext) -> BuildForwardInputsResult:
            # The model gets the inputs. The targets wait in the state until compute_loss().
            return BuildForwardInputsResult(
                input=RegressionInput(x=ctx.batch.x), shared=None, state=RegressionState(y=ctx.batch.y)
            )

        def compute_loss(self, ctx: ComputeLossContext) -> ComputeLossResult:
            loss = F.mse_loss(ctx.pipeline_results, ctx.state.y)
            return ComputeLossResult(loss=loss, loss_weight=None)
    ```

=== "Model Provider"

    ```python
    class MLPProvider(ModelProvider[MLP]):
        def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult[MLP]:
            model = MLP(num_features=16, hidden_size=64)
            return InitializeModelStageResult(model=model, state_mapper=identity_mapper_from_module(model))

        # You choose the parallelism of each submodule. Here, the whole model gets HSDP over the dense mesh domain.
        def parallelize_model_stage(self, context: ParallelizeModelStageContext[MLP]):
            mesh = context.dist_context.mesh_for(DENSE_DOMAIN)
            parallelize_hsdp(context.model, mesh=mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])

        def prepare_export_model_stage(self, context: PrepareExportModelStageContext[MLP]) -> PrepareExportModelStageResult:
            return PrepareExportModelStageResult(state_mapper=identity_mapper_from_module(context.model))
    ```

=== "Run"

    ```python
    config = ProjectConfig.model_validate(yaml.safe_load(Path(sys.argv[1]).read_text()))

    trainer = TrainingConfigurator(
        mesh=config.mesh,
        parameters=config.trainer,
        model_provider=MLPProvider(),
        task_provider=lambda ctx: RegressionTask(),
        data_provider=AutoDataProvider(
            dataset_factory=lambda dist_context: RegressionDataset(num_samples=65536, num_features=16),
            collator=RegressionDataset.collate,
            config=config.data,
        ),
        optimizer_provider=AutoOptimizerProvider(config.optimizer),
        lr_scheduler_provider=AutoLRSchedulerProvider(config.lr_scheduler),
    ).configure()

    trainer.train()
    ```

d9d owns the training loop: gradient synchronization and accumulation, clipping, checkpointing, logging and timeouts.

## Scale Without Changing the Code

To train on 4 GPUs with fully sharded data parallelism, change the mesh in the config and launch the same script with `torchrun`:

```yaml
mesh:
  data_parallel_shard: 4
```

```bash
torchrun --nproc-per-node 4 train.py config.yaml
```

The model provider decides how each submodule is distributed, so the strategy is your code, not a framework setting. See [Choosing Parallelism](./concepts/parallelism.md).

## Extend Without Forking

d9d is a package that you install, not a repository that you fork. Every part of a job is an interface, and your code implements it in your own project. To try a new idea, you write it next to your job and leave d9d untouched:

| To try | You write | See |
|:-------|:----------|:----|
| A new model | An `nn.Module` with `reset_parameters()` and `forward(inputs, shared)`. You can build it from d9d blocks such as attention, MoE and RMSNorm. | [Model Design](./models/model_design.md) |
| A new parallelism strategy | A function that distributes submodules over a device mesh, called from `parallelize_model_stage()`. | [Horizontal Parallelism](./models/horizontal_parallelism.md) |
| A new loss or training method | A `TrainTask`. | [User Tasks](./loop/interfaces/task.md) |
| A new data pipeline | A `DataProvider`. | [Data Loading](./loop/interfaces/data.md) |
| A new optimizer or LR schedule | An `OptimizerProvider` or an `LRSchedulerProvider`. | [Optimizer](./loop/interfaces/optimizer.md) |
| A new checkpoint format | A state mapper. | [Model State Mapper](./model_states/mapper.md) |
| A change to the loop | An event handler, e.g. one that runs RL rollouts between steps. | [Event Bus and Hooks](./loop/interfaces/events.md) |

Updating d9d is then a version bump, not a rebase of your changes.

The same holds for AI coding agents. An agent changes your project, not the framework. It adds a model, a strategy or a task against a small interface and runs the job. There is no fork to patch and keep in sync.

## How d9d Compares

Features converge between training frameworks, so d9d differs in how it is built:

*   **The strategy is your code, not a framework setting**: Other frameworks write the split of each layer into the model code, or apply one mode to the whole model. In d9d, your model provider decides how each submodule is distributed.
*   **The model is yours**: It is a standard `nn.Module` in your project that inherits no framework class and never reads a device mesh. With pipeline parallelism, it builds only the layers of its stage.
*   **Checkpoints are graphs**: Other frameworks convert Hugging Face checkpoints with a hand-written key map per model, or load the whole model on every rank. In d9d, a graph of state mappers streams checkpoints in and out without loading them into memory.
*   **Built on PyTorch primitives**: Parallelism is `DTensor` placements on a `DeviceMesh`, applied with `parallelize_module`, and training checkpoints are PyTorch Distributed Checkpoints. What you know about PyTorch distributed applies directly.
*   **Small components, wired explicitly**: The training loop is a sequence of small components, such as the gradient manager, the clipper and the checkpointer. Each gets its collaborators and the `DistributedContext` through its constructor, with no global process groups and no patches of PyTorch. Configs are Pydantic models, every hook gets a typed context, and all core code passes strict type checking, so you can follow a training step through a few files.

[d9d and Other Frameworks](./concepts/comparison.md) compares these choices with Megatron-Core, torchtitan, DeepSpeed, Accelerate and others, lists the trade-offs of d9d and says when to choose another tool.

## Community

*   **Discord**: [Join the d9d server](https://discord.gg/sNRjDbxVrg) for questions and discussions of distributed training.
*   **Issues**: Report bugs and request features in the [GitHub issue tracker](https://github.com/d9d-project/d9d/issues).
*   **Contributing**: Read [CONTRIBUTING.md](https://github.com/d9d-project/d9d/blob/main/CONTRIBUTING.md) before you open a pull request.
