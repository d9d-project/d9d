# Qwen3 Dense

## About

The `d9d.models.qwen3_dense` package implements the dense [Qwen3](https://arxiv.org/abs/2505.09388) backbone. `parallelize_qwen3_dense_model()` applies the default horizontal parallelism to it: HSDP for all modules. Tensor parallelism and context parallelism are not supported yet.

## Hugging Face Compatibility

The package provides state mappers that convert Hugging Face checkpoints to the d9d layout and back. They use the graph-based [State Mapping](../model_state/mapper.md) engine, so you can pass them to your [Model Provider](../loop/interfaces/model.md).

There is one mapper pair for the bare backbone and one for each head type (causal LM, classification and embedding). An embedding model cannot be exported to Hugging Face if its embedding head has a projection.

## Usage

```python
from d9d.models import DecoderForCausalLM, parallelize_causal_lm_head
from d9d.models.qwen3_dense import (
    Qwen3DenseModel,
    mapper_from_huggingface_qwen3_dense_for_causal_lm,
    parallelize_qwen3_dense_model,
)
from d9d.nn.hidden_states_aggregator import HiddenStatesAggregationMode

# In ModelProvider.initialize_model_stage.
backbone = Qwen3DenseModel(
    params=params,
    stage=context.stage,
    hidden_states_snapshot_mode=HiddenStatesAggregationMode.no,
    enable_checkpointing=True,
)
model = DecoderForCausalLM(backbone, context.stage)
state_mapper = mapper_from_huggingface_qwen3_dense_for_causal_lm(params)

# In ModelProvider.parallelize_model_stage.
parallelize_qwen3_dense_model(context.dist_context, context.model.model, context.stage)
if context.stage.is_current_stage_last:
    parallelize_causal_lm_head(context.model.head, context.dist_context)
```

## API Reference

::: d9d.models.qwen3_dense
