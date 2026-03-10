# Model Definition

## ModelProvider

The `ModelProvider` controls the lifecycle of the `nn.Module`. In distributed training, models are rarely just "instantiated". 

They must be initialized, parallelized, and mapped for loading from checkpoint.

## How to Write a ModelProvider

### Choose a Model
Choose a model from d9d's [catalogue](../models/model_catalogue/index.md) or [create it](../models/model_design.md) by your own.

### Implement `initialize_model_stage(...)`
Implement the `initialize_model_stage(...)` method - it should prepare a `nn.Module` for specified [pipeline parallel](../models/pipeline_parallelism.md) stage containing model architecture in a target `torch.dtype`.

Note that models are initialized on **meta device**, so you **must not** load model weights here.

Instead, this function should return a [State Mapper](../model_states/mapper.md) that will map model weights **on disk** to model weights **in-memory**.

You also may apply [PEFT](../peft/overview.md) methods here and other architectural patches, but make sure you respect the changes they made in returned [State Mapper](../model_states/mapper.md).

### Implement `parallelize_model_stage(...)`
Implement the `parallelize_model_stage(...)` method - it should apply [Horizontal Parallelism](../models/horizontal_parallelism.md) strategy for selected model in-place.

If you use one of d9d's models, you may use default strategies for them such as `parallelize_qwen3_moe_for_causal_lm` ([reference](../models/qwen3_moe.md)).

For a custom model, please see [Horizontal Parallelism](../models/horizontal_parallelism.md) docs and reference implementations.

### Implement `prepare_export_model_stage(...)`
Implement the `prepare_export_model_stage(...)` method - it should return a [State Mapper](../model_states/mapper.md) 
that converts in-memory model state to that one that will be saved on disk during final export.

Basically, it should reverse all the operations of [State Mapper](../model_states/mapper.md) produced in `initialize_model_stage(...)`.

## Example Implementation

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

::: d9d.loop.control.model_provider
