---
title: Interfaces & Logic
---

# Interfaces & Logic

## About

The `d9d` training loop is agnostic to the specific model or data being trained. You interact with the loop by implementing **Providers** (factories) and **Tasks** (step logic).

For standard use cases (like standard Optimizers), `d9d` provides **Auto** implementations that can be configured purely via Pydantic models, avoiding the need to write custom provider classes.

## User Tasks

A **Task** defines custom logic for a single train/inference step.

Each **Task** may implement `Stateful` protocol, so you may store some mutable state here.

### TrainTask

It is responsible for logging metrics, mapping batch inputs before they are fed into the model, and for computing the task loss function value.

**Init**: `create_metrics(...)`, `dump_hparams(...)`.

**Lifecycle**: 

1. `build_forward_inputs(...)` (will be called once) -> 
2. `compute_loss(...)` (will be called multiple times if pipelining is enabled - once for each pipeline microbatch) -> 
3. `update_metrics(...)` (will be called once).

**Exit**: `finalize(...)`.

**State Management**: `state_dict(...)`, `load_state_dict(...)`.

### InferenceTask

The `InferenceTask` defines the logic for a single inference step. 

It is designed to handle the **forward-only** flow, processing the raw tensors synthesized by the model (e.g., logits, hidden states).

**Lifecycle**: 

1. `build_forward_inputs(...)` (called once) -> 
2. `process_outputs(...)` (called once per pipeline microbatch).

**Exit**: `finalize(...)`.

**State Management**: `state_dict(...)`, `load_state_dict(...)`.

### Pipeline State

You may note that `batch` is only accessible in `build_forward_inputs(...)` method, but not in others. Don't worry!

There is an object for transferring any state between the **Task Lifecycle** stages, - it is called `PipelineState`.

```python
ctx.state["target"] = torch.tensor([1, 0, 1, 0], device="cuda")

# ...

metrics["accuracy"].update(ctx.state["target"])
```

The pipeline state will automatically shard and unshard data if needed.

You may read an [additional documentation](../internals/pipeline_state.md) for its internal behaviour.

### Example Implementation

```python
import torch

from d9d.core.dist_context import DistributedContext
from d9d.core.types import ScalarTree
from d9d.module.block.head import LM_IGNORE_INDEX
from d9d.loop.control import *

class SFTTask(TrainTask[dict[str, torch.Tensor]]):
    def __init__(self, dist_ctx: DistributedContext):
        self._dist_ctx = dist_ctx

    def build_forward_inputs(self, ctx: BuildForwardInputsContext) -> BuildForwardInputsResult:
        # ctx.batch contains the output of the Collator.

        # Save labels in state for access during loss computation later
        ctx.state["labels"] = ctx.batch["labels"]

        # Return inputs for model.forward()
        # inputs are only for the first pipeline stage
        # kwargs are the same for all the pipeline stages
        return BuildForwardInputsResult(
            inputs={
                "input_ids": ctx.batch["input_ids"]
            },
            kwargs={
                "labels": ctx.batch["labels"],
                "position_ids": ctx.batch["position_ids"]
            }
        )

    def dump_hparams(self) -> ScalarTree:
        return super().dump_hparams()

    def compute_loss(self, ctx: ComputeLossContext) -> ComputeLossResult:
        # Retrieve log_probs calculated by the model pipeline
        logps = ctx.pipeline_results["logps"]

        # Calculate number of valid tokens (ignoring the -100 padding)
        # This is crucial for variable length batches.
        num_loss_tokens = (ctx.state["labels"] != LM_IGNORE_INDEX).sum()

        # Calculate average loss per valid token
        total_loss = logps.sum() / num_loss_tokens

        return ComputeLossResult(
            loss=total_loss,
            # loss_weight is used for gradient accumulation across the distributed world.
            # If batches have different token counts, we weigh the gradient
            # by token count to get a mathematical true average over the accumulation steps.
            loss_weight=num_loss_tokens / 1000
        )
```

### API Reference

::: d9d.loop.control.task
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3

## Model Definition

### ModelProvider

The `ModelProvider` controls the lifecycle of the `nn.Module`. In distributed training, models are rarely just "instantiated". 

They must be initialized, parallelized, and mapped for loading from checkpoint.

### How to Write a ModelProvider

#### Choose a Model
Choose a model from d9d's [catalogue](../models/4_model_catalogue.md) or [create it](../models/1_model_design.md) by your own.

#### Implement `initialize_model_stage(...)`
Implement the `initialize_model_stage(...)` method - it should prepare a `nn.Module` for specified [pipeline parallel](../models/3_pipeline_parallelism.md) stage containing model architecture in a target `torch.dtype`.

Note that models are initialized on **meta device**, so you **must not** load model weights here.

Instead, this function should return a [State Mapper](../model_states/mapper.md) that will map model weights **on disk** to model weights **in-memory**.

You also may apply [PEFT](../peft/0_index.md) methods here and other architectural patches, but make sure you respect the changes they made in returned [State Mapper](../model_states/mapper.md).

#### Implement `parallelize_model_stage(...)`
Implement the `parallelize_model_stage(...)` method - it should apply [Horizontal Parallelism](../models/2_horizontal_parallelism.md) strategy for selected model in-place.

If you use one of d9d's models, you may use default strategies for them such as `parallelize_qwen3_moe_for_causal_lm` ([reference](../models/qwen3_moe.md)).

For a custom model, please see [Horizontal Parallelism](../models/2_horizontal_parallelism.md) docs and reference implementations.

#### Implement `prepare_export_model_stage(...)`
Implement the `prepare_export_model_stage(...)` method - it should return a [State Mapper](../model_states/mapper.md) 
that converts in-memory model state to that one that will be saved on disk during final export.

Basically, it should reverse all the operations of [State Mapper](../model_states/mapper.md) produced in `initialize_model_stage(...)`.

### Example Implementation

```python
from pydantic import BaseModel
from d9d.loop.control.model_provider import *
from d9d.module.model.qwen3_moe import Qwen3MoEForCausalLM, Qwen3MoEForCausalLMParameters
from d9d.module.parallelism.model.qwen3_moe import parallelize_qwen3_moe_for_causal_lm
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.model_state.mapper.adapters import identity_mapper_from_module


class ModelProviderConfig(BaseModel):
    model: Qwen3MoEForCausalLMParameters  # Hyperparameters for Qwen3 MoE
    checkpointing: bool  # Enable gradient checkpointing to save VRAM


class ProjectModelProvider(ModelProvider[Qwen3MoEForCausalLM]):
    def __init__(self, config: ModelProviderConfig):
        self._config = config

    def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult:
        # Initialize the raw model on Meta device in BF16 precision
        model = Qwen3MoEForCausalLM(
            params=self._config.model,
            stage=context.stage,
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=self._config.checkpointing
        ).bfloat16()

        return InitializeModelStageResult(
            model=model,
            state_mapper=identity_mapper_from_module(model)
        )

    def parallelize_model_stage(self, context: ParallelizeModelStageContext):
        # Applies specific distributed strategies
        # suited for Qwen3 MoE architecture.
        # You can apply your own horizontal parallelism strategy here.
        parallelize_qwen3_moe_for_causal_lm(
            dist_context=context.dist_context,
            stage=context.stage,
            model=context.model
        )

    def prepare_export_model_stage(self, context: PrepareExportModelStageContext) -> PrepareExportModelStageResult:
        # When exporting, save model weights as-is

        return PrepareExportModelStageResult(
            state_mapper=identity_mapper_from_module(context.model)
        )

    def dump_hparams(self) -> ScalarTree:
        return self._config.model_dump(mode="json")
```

### API Reference

::: d9d.loop.control.model_provider
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3

## Data Loading

### DatasetProvider

The `DatasetProvider` is responsible for creating dataset and data collator instances.

### Distributed-Awareness

d9d **will not** apply sharding to your dataset automatically. You have to configure it manually (optionally applying other dataset wrappers).

Please see the [Dataset Utilities](../dataset/index.md) documentation.

### Example Implementation

```python
from typing import Any, Sequence

import torch
import datasets
from pydantic import BaseModel
from tokenizers import Tokenizer

from d9d.core.types import TensorTree
from d9d.dataset import BufferSortedDataset, shard_dataset_data_parallel, DatasetImplementingSortKeyProtocol
from d9d.loop.control.dataset_provider import *

class ProjectDataset(Dataset, DatasetImplementingSortKeyProtocol):
    def __init__(self, dataset: datasets.Dataset, tokenizer: Tokenizer):
        self._dataset = dataset
        self._tokenizer = tokenizer

    def sort_key(self, index: int) -> Any:
        # Used by BufferSortedDataset to group examples of similar length together.
        # This minimizes padding overhead in batches.
        return self._dataset[index]["token_counts"]

    def __getitem__(self, index: int) -> TensorTree:
        return {
            ...
        }

    @classmethod
    def collate(cls, batch: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            ...
        }

    def __len__(self) -> int:
        return len(self._dataset)


class DataConfig(BaseModel):
    dataset: str  # HuggingFace dataset path/name
    split: str  # e.g., 'train', 'validation'
    text_column: str  # The column containing the raw text
    use_samples: int  # Limit dataset size for testing/debugging
    shuffle_seed: int  # Distinct seed for shuffling the data
    tokenizer: str  # Path to the tokenizer.json file
    num_proc: int  # Number of CPU processes for data mapping
    presort_buffer_size: int  # Size of buffer for length-based presorting

    
class ProjectDatasetProvider(DatasetProvider):
    def __init__(self, config: DataConfig):
        self._config = config

    @staticmethod
    def _count_tokens(item: dict, text_column: str, tokenizer: Tokenizer) -> dict:
        return {
            "token_counts": len(tokenizer.encode(item[text_column]).tokens)
        }

    def __call__(self, context: InitializeDatasetContext) -> InitializeDatasetResult:
        tokenizer = Tokenizer.from_file(str(self._config.tokenizer))
        # IMPORTANT: main_process_first ensures that Rank 0 downloads/processes
        # the dataset and builds the cache first. Ranks 1-N wait, then load from cache.
        # Prevents race conditions and corruption on the HF cache.
        with context.dist_context.main_process_first():
            data = datasets.load_dataset(
                self._config.dataset,
                split=self._config.split
            ).take(
                self._config.use_samples
            ).shuffle(
                self._config.shuffle_seed
            ).map(
                self._count_tokens,
                num_proc=self._config.num_proc,
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "text_column": self._config.text_column
                }
            )

        dataset = ProjectDataset(data, tokenizer)

        # BufferSortedDataset acts as a buffer that shuffles data locally
        # but outputs batches sorted by length (defined in sort_key above)
        dataset_buf = BufferSortedDataset(
            dataset,
            buffer_size=self._config.presort_buffer_size,
            pack_size=context.batch_maths.global_batch_size,
            init_seed=self._config.shuffle_seed
        )

        # Split dataset across data parallel ranks
        dataset_shard = shard_dataset_data_parallel(dataset_buf, context.dist_context)

        return InitializeDatasetResult(
            dataset=dataset_shard,
            collator=ProjectDataset.collate
        )
```

### API Reference

::: d9d.loop.control.dataset_provider
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3

## Optimization & Scheduling

### Auto Implementations

For standard PyTorch usage, `d9d` includes the `d9d.loop.auto` package. These providers ingest a Pydantic configuration object and manage the creation of standard optimizers and schedulers.

#### Auto Optimizer

Supports `AdamW`, `Adam`, `SGD`, and `StochasticAdamW`.

```python
from d9d.loop.auto import AutoOptimizerProvider, AutoOptimizerConfig

provider = AutoOptimizerProvider(
    AutoOptimizerConfig.model_validate_json('{"name": "adamw", "lr": 1e-4}')
)
```

::: d9d.loop.auto.auto_optimizer
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3

#### Auto Scheduler

Supports [Piecewise Linear](../lr_scheduler/piecewise.md) schedules (warmup, hold, decay).

```python
from d9d.loop.auto import AutoLRSchedulerProvider, AutoLRSchedulerConfig

cfg = """
{
    "initial_multiplier": 0.0,
    "phases": [
        {
            "mode": "steps",
            "steps": 100,
            "target_multiplier": 1.0,
            "curve": { "type": "linear" }
        },
        {
            "mode": "rest",
            "target_multiplier": 0.1,
            "curve": { "type": "cosine" }
        }
    ]
}
"""

provider = AutoLRSchedulerProvider(
    AutoLRSchedulerConfig.model_validate_json(cfg)
)
```

::: d9d.loop.auto.auto_lr_scheduler
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3

### Interface

If you need a custom optimizer or learning rate scheduler, you implement the `OptimizerProvider` protocol.

::: d9d.loop.control.optimizer_provider
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3

::: d9d.loop.control.lr_scheduler_provider
    options:
        show_root_heading: true
        show_root_full_path: true
        heading_level: 3
