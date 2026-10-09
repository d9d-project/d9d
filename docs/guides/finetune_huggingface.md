# Fine-Tune a Hugging Face Model

## About

This guide loads [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) from its Hugging Face checkpoint, fine-tunes it on [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext) and exports it back in the Hugging Face format. A second config trains LoRA adapters instead of all weights. Another script measures the perplexity of a model with the inference loop. The code is in [`example/guide/finetune_qwen3`](https://github.com/d9d-project/d9d/tree/main/example/guide/finetune_qwen3).

!!! warning "Hugging Face integration will change"
    d9d will integrate with Hugging Face more closely ([#69](https://github.com/d9d-project/d9d/issues/69)), so this guide will change. Today, you copy the model parameters from `config.json` by hand and copy the config and the tokenizer files after export.

## Run the Example

Install d9d and get the repository as in the [Quickstart](../getting_started/quickstart.md). The example also reads the data with the `datasets` and `tokenizers` packages:

```bash
pip install datasets tokenizers
cd d9d/example/guide/finetune_qwen3
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
python finetune.py config.yaml
```

The job trains for 100 steps on one GPU. It saves a checkpoint to `runs/checkpoints/finetune-qwen3/` and the exported model to `runs/export/`.

## Model Parameters

The `model` section of the config holds the `Qwen3DenseParameters`. Copy them from the `config.json` file of the Hugging Face model:

| Hugging Face `config.json` | d9d `model` |
|:---------------------------|:------------|
| `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `rms_norm_eps` | `layer.<same name>` |
| `num_hidden_layers` | `num_hidden_layers` |
| `rope_theta` | `rope_base` |
| `max_position_embeddings` | `max_position_ids` |
| `vocab_size` | `split_vocab_size.regular`, with `split_vocab_order` set to `["regular"]` |

For Qwen3-0.6B, the section is:

```yaml
--8<-- "example/guide/finetune_qwen3/config.yaml:model"
```

The Hugging Face mappers support only one vocabulary split.

## Loading the Hugging Face Model

`trainer.model_stage_factory.source_checkpoint` points to the downloaded model directory:

```yaml
trainer:
  model_stage_factory:
    source_checkpoint: models/Qwen3-0.6B
```

The job streams the weights into the model through the state mapper that `initialize_model_stage()` returns (see [The Model Build](../concepts/how_d9d_works.md#the-model-build)). The model provider also handles LoRA and export, which the sections below describe:

```python
--8<-- "example/guide/finetune_qwen3/finetune.py:provider"
```

See [Model State Mapper](../model_states/mapper.md).

## Optimizer

The provider casts the model to bf16 with `.bfloat16()`. The config keeps the gradients in fp32 and uses `stochastic_adamw`, which updates bf16 weights with stochastic rounding (see [Stochastic Optimizers](../optimizer/stochastic.md)):

```yaml
trainer:
  gradient_manager:
    grad_dtype: float32

optimizer:
  name: stochastic_adamw
  lr: 1.0e-05
  state_dtype: float32
```

The `adamw` optimizer requires the gradients to have the dtype of the weights, so it fails with this config.

## Loss

The causal LM head computes the loss itself, so the task passes the labels to it in `shared` and keeps only the number of label tokens in the state:

```python
--8<-- "example/guide/finetune_qwen3/finetune.py:task"
```

Returning the number of tokens as the `loss_weight` makes the loop average the loss per token across microbatches and ranks.

## Export to Hugging Face

`prepare_export_model_stage()` of the [provider](#loading-the-hugging-face-model) returns `mapper_to_huggingface_qwen3_dense_for_causal_lm`, so `Trainer.export()` writes the weights with the Hugging Face keys. The export has only the `.safetensors` files. Copy the config and the tokenizer from the original model to load it with Transformers:

```bash
cp models/Qwen3-0.6B/{config.json,generation_config.json,tokenizer.json,tokenizer_config.json} runs/export/
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("runs/export")
model = AutoModelForCausalLM.from_pretrained("runs/export", dtype="bfloat16").cuda()

inputs = tokenizer("The history of the Roman Empire", return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=25)[0]))
```

Qwen3-0.6B ties its input embeddings to its LM head. d9d trains them as two separate weights, so Transformers logs that it does not tie them. Set `"tie_word_embeddings": false` in the copied `config.json` to silence the message.

## Fine-Tune with LoRA

`config_lora.yaml` adds a `lora` section, which trains [LoRA](../peft/lora.md) adapters on the attention projections:

```yaml
lora:
  module_name_pattern: .*self_attn\.(q|k|v|o)_proj
  params:
    r: 16
    alpha: 32
    dropout: 0.0
```

```bash
python finetune.py config_lora.yaml
```

The [provider](#loading-the-hugging-face-model) changes in three places:

*   **Injection**: `initialize_model_stage()` wraps the matching layers and freezes all other weights. It chains the LoRA mapper after the Hugging Face mapper, because the wrapped layers move their weights to new keys.
*   **Checkpoints**: The config sets `checkpoint_only_trainable_parameters`, so the checkpoints hold only the adapters: 46 MiB instead of 7.1 GiB.
*   **Export**: `prepare_export_model_stage()` merges the adapters into the base weights first. The export in `runs/export-lora/` has the original architecture. Copy the config and the tokenizer into it as above to load it with Transformers.

## Evaluate the Model

`evaluate.py` computes the perplexity on the validation split with the [inference loop](../loop/inference.md). It takes the model state to evaluate as its second argument. It loads it through the same Hugging Face mapper, so it reads the original model and the exports alike:

```bash
python evaluate.py evaluate.yaml models/Qwen3-0.6B
python evaluate.py evaluate.yaml runs/export
python evaluate.py evaluate.yaml runs/export-lora
```

Each run logs the mean loss and the perplexity on the WikiText-2 validation split.

The script reuses the dataset and the model provider of `finetune.py`. Its task sums the loss and the number of tokens of each microbatch and computes the perplexity at the end:

```python
--8<-- "example/guide/finetune_qwen3/evaluate.py:task"
```

## More GPUs

`parallelize_model_stage()` applies the default Qwen3 strategy: HSDP over the data-parallel dimensions. Set `data_parallel_replicate` or `data_parallel_shard` in `mesh` and launch with `torchrun`, as described in [Running Jobs](./running.md). See [Qwen3 Dense](../models/model_catalogue/qwen3_dense.md) for the supported parallelism.

## The Code

=== "finetune.py"

    ```python
    --8<-- "example/guide/finetune_qwen3/finetune.py"
    ```

=== "evaluate.py"

    ```python
    --8<-- "example/guide/finetune_qwen3/evaluate.py"
    ```
