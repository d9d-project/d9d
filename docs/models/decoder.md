# Decoder with Heads

## About

The `d9d.models` package composes a decoder backbone, such as [Qwen3 Dense](./qwen3_dense.md), with [task heads](../nn/head.md) into one model that supports pipeline parallelism. The `d9d.models.io` package holds the types that this model passes between pipeline stages.

## Composition

The backbone maps token ids to hidden states. A head turns the hidden states into the output of a task, e.g. a loss or logits. The model attaches the heads only on the last pipeline stage.

*   **One head**: `DecoderWithHead` stores the head as `head` and returns its output as it is. `DecoderForCausalLM`, `DecoderForClassification` and `DecoderForEmbedding` build the head themselves.
*   **Several heads**: `DecoderWithHeads` stores the heads in `heads` by name and returns a mapping from the name to the output. Each head gets its own shared input under its name.

`build_decoder_head` builds a built-in head from its configuration. It takes the hidden size and the vocabulary splits from the backbone. You can also pass your own `TaskHead` to the model.

The backbone and the heads are public attributes. So the model provider parallelizes each of them separately, e.g. with `parallelize_causal_lm_head`, and the state mappers address them by their prefixes `model.`, `head.` and `heads.<name>.`.

## Usage

```python
from d9d.models import (
    CausalLMHeadConfig,
    ClassificationHeadConfig,
    DecoderWithHeads,
    build_decoder_head,
    parallelize_causal_lm_head,
    parallelize_classification_head,
)

# In ModelProvider.initialize_model_stage.
backbone = ...  # E.g. a Qwen3DenseModel.
heads = {
    "lm": build_decoder_head(CausalLMHeadConfig(), backbone=backbone),
    "cls": build_decoder_head(ClassificationHeadConfig(num_labels=2), backbone=backbone),
}
model = DecoderWithHeads(backbone, heads, context.stage)

# In ModelProvider.parallelize_model_stage, after the backbone.
if context.stage.is_current_stage_last:
    parallelize_causal_lm_head(context.model.heads["lm"], context.dist_context)
    parallelize_classification_head(context.model.heads["cls"], context.dist_context)
```

## API Reference

::: d9d.models

::: d9d.models.io
