# Write Your Own Job

## About

This page turns the [Quickstart](../getting_started/quickstart.md) script into a job for your own model and data. Copy `train.py` and `config.yaml` from `example/quickstart` into your project and replace their parts in the order below. Before each run, delete `runs/` or change `trainer.run.name`. Otherwise, the job tries to resume the finished Quickstart run with your new code.

## Replace the Data

Replace `RegressionDataset` with your `Dataset` and its `collate()` with a function that stacks samples into one microbatch. The task receives the microbatch as `ctx.batch`.

`dataset_factory` runs on every rank. Return the whole dataset from it: `AutoDataProvider` shards it across the data-parallel ranks. Wrap downloads in `dist_context.main_process_first()`, so that only rank 0 fills the cache. See [Data Loading](../loop/interfaces/data.md#using-autodataprovider).

If samples differ in length, pad them in `collate()` with [`pad_stack_1d`](../dataset/index.md#padding) and mark the padding in the labels. The task then weights the loss by the number of real tokens (see [Write the Task](#write-the-task)). The [Qwen3 fine-tuning example](./finetune_huggingface.md) does both:

```python
# One microbatch of token sequences, padded to one length.
@dataclasses.dataclass
class TextBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    position_ids: torch.Tensor


class TextDataset(Dataset):
    ...

    @staticmethod
    def collate(samples: Sequence[torch.Tensor]) -> TextBatch:
        # d9d models do not shift the labels, so shift them here.
        return TextBatch(
            input_ids=pad_stack_1d([tokens[:-1] for tokens in samples], pad_value=0),
            labels=pad_stack_1d([tokens[1:] for tokens in samples], pad_value=LM_IGNORE_INDEX),
            position_ids=pad_stack_1d([torch.arange(len(tokens) - 1) for tokens in samples], pad_value=0),
        )


def build_dataset(config: DataConfig, dist_context: DistributedContext) -> Dataset:
    # Rank 0 downloads the dataset first. The other ranks then read it from the cache.
    with dist_context.main_process_first():
        data = datasets.load_dataset(config.dataset, config.dataset_config, split=config.split)
    texts = [text for text in data[config.text_column] if text.strip()]
    return TextDataset(texts, Tokenizer.from_file(str(config.tokenizer)), config.max_length)


# The data provider of TrainingConfigurator. It calls build_dataset() once on every rank
# and TextDataset.collate() for every microbatch.
data_provider = AutoDataProvider(
    dataset_factory=lambda dist_context: build_dataset(config.data, dist_context),
    collator=TextDataset.collate,
    config=config.auto_data,
)
```

`AutoDataProvider` is optional. For streaming data, your own batching or a varying number of microbatches per step, write your own `DataProvider` that returns a pack stream. Then you also shard the data yourself. See [Writing a Custom DataProvider](../loop/interfaces/data.md#writing-a-custom-dataprovider).

## Replace the Model

Replace `MLP` with your `nn.Module`. It needs `forward(inputs, shared)` and `reset_parameters()`. `inputs` and `shared` are what the task returns as `input` and `shared` from `build_forward_inputs()` (see [Write the Task](#write-the-task)).

d9d builds the model on the `meta` device, allocates uninitialized memory for it on the GPU and calls only the `reset_parameters()` of the model. So this method must initialize every parameter and buffer of every submodule:

```python
class Block(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.norm(x))

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.linear.reset_parameters()


class ResidualMLP(nn.Module):
    def __init__(self, num_features: int, hidden_size: int, num_blocks: int):
        super().__init__()
        self.up = nn.Linear(num_features, hidden_size)
        self.blocks = nn.ModuleList(Block(hidden_size) for _ in range(num_blocks))
        self.down = nn.Linear(hidden_size, 1)
        self.register_buffer("input_scale", torch.empty(num_features))

    def forward(self, inputs: RegressionInput, shared: None) -> torch.Tensor:
        x = self.up(inputs.x * self.input_scale)
        for block in self.blocks:
            x = block(x)
        return self.down(x)

    def reset_parameters(self):
        self.up.reset_parameters()
        for block in self.blocks:
            block.reset_parameters()
        self.down.reset_parameters()
        self.input_scale.fill_(1.0)
```

See [Model Design](../models/model_design.md).

## Write the Model Provider

Replace `MLPProvider` with a provider for your model and pass it to `TrainingConfigurator` as `model_provider`. This provider for the `ResidualMLP` above reads the sizes of the model from a config section:

```python
class ModelConfig(BaseModel):
    num_features: int
    hidden_size: int
    num_blocks: int


class ResidualMLPProvider(ModelProvider[ResidualMLP]):
    def __init__(self, config: ModelConfig):
        self._config = config

    # Builds the model and returns the state mapper for loading.
    def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult[ResidualMLP]:
        model = ResidualMLP(
            num_features=self._config.num_features,
            hidden_size=self._config.hidden_size,
            num_blocks=self._config.num_blocks,
        )
        # The model trains from scratch and keeps its own parameter names, so the mapper changes nothing.
        return InitializeModelStageResult(model=model, state_mapper=identity_mapper_from_module(model))

    # Distributes the model. The loop calls it only in a distributed run, not on one GPU.
    def parallelize_model_stage(self, context: ParallelizeModelStageContext[ResidualMLP]):
        mesh = context.dist_context.mesh_for(DENSE_DOMAIN)
        parallelize_hsdp(context.model, mesh=mesh["dp_replicate", "dp_cp_shard", "cp_replicate"])

    # Returns the state mapper for export.
    def prepare_export_model_stage(
        self, context: PrepareExportModelStageContext[ResidualMLP]
    ) -> PrepareExportModelStageResult:
        return PrepareExportModelStageResult(state_mapper=identity_mapper_from_module(context.model))


class ProjectConfig(BaseModel):
    model: ModelConfig
    ...


# The model provider of TrainingConfigurator in main().
model_provider = ResidualMLPProvider(config.model)
```

To start from pretrained weights, set `trainer.model_stage_factory.source_checkpoint`. The loop then loads the checkpoint through the [state mapper](../model_states/mapper.md) of `initialize_model_stage()`, which translates its keys into the keys of your model. The [Qwen3 fine-tuning example](./finetune_huggingface.md#loading-the-checkpoint) loads a Hugging Face checkpoint this way. See [Model Definition](../loop/interfaces/model.md).

## Write the Task

Replace `RegressionTask` with your loss. The task connects the data and the model. For each microbatch, the loop calls `build_forward_inputs()` before the forward pass and `compute_loss()` on the output of the last pipeline stage. The state carries data from the first call to the second, so the model receives only what it computes on.

This is the task of the Quickstart:

```python
# The type parameters describe the microbatch, input, shared, the model output and state. They serve only
# type checking. Each part can be a dataclass, a dict, or None if the job has no such data.
class RegressionTask(TrainTask[RegressionBatch, RegressionInput, None, torch.Tensor, RegressionState]):
    # Gets the microbatch that collate() built and splits it into three parts.
    def build_forward_inputs(
        self, ctx: BuildForwardInputsContext[RegressionBatch]
    ) -> BuildForwardInputsResult[RegressionInput, None, RegressionState]:
        return BuildForwardInputsResult(
            # The model receives it as `inputs`.
            input=RegressionInput(x=ctx.batch.x),
            # The model receives it as `shared`. With pipeline parallelism, every stage receives it.
            shared=None,
            # What compute_loss() and update_metrics() need but the model does not return.
            state=RegressionState(y=ctx.batch.y),
        )

    # Gets the model output and the state of the same microbatch.
    def compute_loss(self, ctx: ComputeLossContext[torch.Tensor, RegressionState]) -> ComputeLossResult:
        loss = nn.functional.mse_loss(ctx.pipeline_results, ctx.state.y)
        # The loop averages the losses of all microbatches and ranks, weighted by loss_weight. None means 1.
        return ComputeLossResult(loss=loss, loss_weight=None)


# The task provider of TrainingConfigurator in main(). The loop calls it after it builds the distributed
# context and passes the context in ctx.dist_context.
task_provider = lambda ctx: RegressionTask()
```

For a loss per token, return the number of tokens as `loss_weight`, as in the [Qwen3 fine-tuning task](./finetune_huggingface.md#loss). See [User Tasks](../loop/interfaces/task.md).

## Add Metrics

The loop logs the loss and the gradient norm without any code. For other values, add metrics to the task. The loop aggregates them across the ranks, logs them every `logging.period_steps` steps and then resets them. This metric counts the samples of the Quickstart task:

```python
from d9d.metric.impl.aggregation import SumMetric


class RegressionTask(TrainTask[...]):
    ...

    # Called once, when the loop builds the job. The keys are the names in the log.
    def create_metrics(self, ctx: CreateMetricsContext) -> CreateMetricsResult:
        return CreateMetricsResult(metrics={"num_samples": SumMetric()})

    # Called after compute_loss() for each microbatch. It gets the state, not the model output, so
    # build_forward_inputs() must put everything that a metric needs into the state.
    def update_metrics(self, ctx: UpdateMetricsContext[RegressionState]):
        ctx.metrics["num_samples"].update(torch.ones_like(ctx.state.y))
```

With the Quickstart config, the log shows `num_samples=2560` every 10 steps. The [Metric Catalogue](../metric/metric_catalogue/index.md) lists the ready metrics, and [Custom Metrics](../metric/custom.md) shows how to write your own. See [Metrics Overview](../metric/overview.md).

## Set the Config

Set `trainer.run.name` and the batch sizes in `data`. On one GPU, `mesh` stays empty. To run on several GPUs, see [Running Jobs](./running.md). [Configuration](../loop/config.md) describes the other fields.

Add a section for each config model that you add to `ProjectConfig`, e.g. the `model` section of the provider above:

```yaml
model:
  num_features: 16
  hidden_size: 64
  num_blocks: 2
```

The [Qwen3 fine-tuning example](./finetune_huggingface.md) has `model`, `lora` and `data` sections this way.
