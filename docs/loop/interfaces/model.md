# Model Definition

## About

The `ModelProvider` controls the lifecycle of the `nn.Module`. In distributed training, a model is not just created. It must be initialized, parallelized and mapped for loading from a checkpoint.

## How to Write a ModelProvider

### Choose a Model

Choose a model from the d9d [catalogue](../../models/model_catalogue/index.md), or [create your own](../../models/model_design.md).

### Implement `initialize_model_stage(...)`

This method builds the `nn.Module` for one [pipeline parallel](../../models/pipeline_parallelism.md) stage, in the target `torch.dtype`.

The loop calls it on the **meta device**, so you **must not** load model weights here. Instead, return a [state mapper](../../model_states/mapper.md) that maps the weights **on disk** to the weights **in memory**.

You can also apply [PEFT](../../peft/overview.md) methods and other architecture patches here. The returned state mapper must reflect the changes they make.

### Implement `parallelize_model_stage(...)`

This method applies a [horizontal parallelism](../../models/horizontal_parallelism.md) strategy to the model in place. The loop calls it only for distributed runs.

For d9d models, you can use the default strategies. For example, apply `parallelize_qwen3_moe_model` to the backbone and the routine for each head type, such as `parallelize_causal_lm_head` ([reference](../../models/model_catalogue/qwen3_moe.md)).

For a custom model, see the [horizontal parallelism](../../models/horizontal_parallelism.md) docs and the reference implementations.

### Implement `prepare_export_model_stage(...)`

This method returns a [state mapper](../../model_states/mapper.md) for the final export. It converts the in-memory model state to the state saved on disk.

It usually reverses the state mapper returned by `initialize_model_stage(...)`.

## Usage

```python
from pydantic import BaseModel

from d9d.core.types import ScalarTree
from d9d.loop.control import (
    InitializeModelStageContext,
    InitializeModelStageResult,
    ModelProvider,
    ParallelizeModelStageContext,
    PrepareExportModelStageContext,
    PrepareExportModelStageResult,
)
from d9d.model_state.mapper.adapters import identity_mapper_from_module
from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model import DecoderForCausalLM
from d9d.module.model.qwen3_moe import Qwen3MoEModel, Qwen3MoEParameters
from d9d.module.parallelism.model import parallelize_causal_lm_head, parallelize_qwen3_moe_model


class ModelProviderConfig(BaseModel):
    model: Qwen3MoEParameters  # Hyperparameters for the Qwen3 MoE backbone
    checkpointing: bool  # Enable activation checkpointing to save GPU memory


class ProjectModelProvider(ModelProvider[DecoderForCausalLM[Qwen3MoEModel]]):
    def __init__(self, config: ModelProviderConfig):
        self._config = config

    def initialize_model_stage(self, context: InitializeModelStageContext) -> InitializeModelStageResult:
        # Compose the Qwen3 MoE backbone with a single causal LM head and cast it to bf16.
        backbone = Qwen3MoEModel(
            params=self._config.model,
            stage=context.stage,
            hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
            enable_checkpointing=self._config.checkpointing,
        )
        model = DecoderForCausalLM(backbone, context.stage).bfloat16()

        return InitializeModelStageResult(
            model=model,
            state_mapper=identity_mapper_from_module(model),
        )

    def parallelize_model_stage(self, context: ParallelizeModelStageContext):
        # Apply the Qwen3 MoE parallelism routine to the backbone and the head routine to the head.
        # You can apply your own horizontal parallelism strategy here.
        parallelize_qwen3_moe_model(context.dist_context, context.model.model, context.stage)
        if context.stage.is_current_stage_last:
            parallelize_causal_lm_head(context.model.head, context.dist_context)

    def prepare_export_model_stage(self, context: PrepareExportModelStageContext) -> PrepareExportModelStageResult:
        # Export the model weights as they are.
        return PrepareExportModelStageResult(
            state_mapper=identity_mapper_from_module(context.model),
        )

    def dump_hparams(self) -> ScalarTree:
        return self._config.model_dump(mode="json")
```

## API Reference

::: d9d.loop.control.model_provider
