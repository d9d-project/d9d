# Qwen3 Dense

## About

The `d9d.module.model.qwen3_dense` package implements the dense [Qwen3](https://arxiv.org/abs/2505.09388) backbone. The `d9d.module.parallelism.model.qwen3_dense` package applies the default horizontal parallelism to it: HSDP for all modules. Tensor parallelism and context parallelism are not supported yet.

## Hugging Face Compatibility

The package provides state mappers that convert Hugging Face checkpoints to the d9d layout and back. They use the graph-based [State Mapping](../../model_states/mapper.md) engine, so you can pass them to your [Model Provider](../../loop/interfaces/model.md).

There is one mapper pair for the bare backbone and one for each head type (causal LM, classification and embedding). An embedding model cannot be exported to Hugging Face if its embedding head has a projection.

`parameters_from_huggingface_qwen3_dense` builds the `Qwen3DenseParameters` from the `config.json` of a Hugging Face model. It accepts the layouts of Transformers 4 and Transformers 5. It raises an error if the config enables a feature that the model does not support, e.g. sliding window attention, RoPE scaling or attention bias.

## Usage

```python
import json
from pathlib import Path

from d9d.module.block.hidden_states_aggregator import HiddenStatesAggregationMode
from d9d.module.model import DecoderForCausalLM
from d9d.module.model.qwen3_dense import (
    Qwen3DenseModel,
    mapper_from_huggingface_qwen3_dense_for_causal_lm,
    parameters_from_huggingface_qwen3_dense,
)
from d9d.module.parallelism.model import parallelize_causal_lm_head, parallelize_qwen3_dense_model

hf_config = json.loads(Path("models/Qwen3-0.6B/config.json").read_text())
params = parameters_from_huggingface_qwen3_dense(hf_config, vocab_split_name="regular")

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

::: d9d.module.model.qwen3_dense

::: d9d.module.parallelism.model.qwen3_dense
