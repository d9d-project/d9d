---
DEP: 
Title: Model Parameters from Hugging Face Configs
Author: Aleksei Makarchik @instapilchik
Status: Draft
Type: Feature
Created: 2026-10-14
---

# DEP-XXXX: Model Parameters from Hugging Face Configs

## Abstract

Users who fine-tune a catalogue model copy its architecture from the Hugging Face `config.json` into d9d model parameters by hand. This DEP adds two layers per model family. A converter function builds the family's parameters from a Hugging Face config and rejects the architecture options that the d9d model does not implement. A closed Pydantic discriminated union selects where the parameters come from: written out explicitly or read from a local Hugging Face model directory. A `build_*_parameters()` factory resolves the union with an exhaustive `match`, following the pattern of DEP-0008. This DEP implements both layers for Qwen3 Dense. Qwen3 MoE follows the same pattern in a separate PR. Export to Hugging Face is out of scope.

## Motivation

Issue #69 asks for closer Hugging Face integration. The input side has three problems today.

*   **Manual copy.** The config of the [fine-tuning guide](../docs/guides/finetune_huggingface.md) repeats 10 fields of the Qwen3-0.6B `config.json` under other names, e.g. `rope_theta` becomes `rope_base`, `max_position_embeddings` becomes `max_position_ids` and `vocab_size` becomes `split_vocab_size`. Each user repeats this for each model.
*   **No check of the copy.** Nothing compares the copied parameters with the Hugging Face config. A wrong `rope_theta` does not change any weight shape, so the checkpoint loads and the job trains a different model. If the source model enables an option that the d9d model lacks, such as sliding window attention or attention bias, nothing reports it.
*   **Two config layouts.** Transformers 4 writes `rope_theta` and `rope_scaling`. Transformers 5 writes `rope_parameters` and `layer_types`. Configs on the Hub are saved by both versions, so a parser has to read both layouts.

## Design Proposal

### Layer 1: the converter function

Each family gets a function in its `huggingface.py`, next to the state mappers:

```python
def parameters_from_huggingface_qwen3_dense(
    config: Mapping[str, Any], *, vocab_split_name: str
) -> Qwen3DenseParameters: ...
```

The function validates the parsed `config.json` with a private Pydantic model and maps the fields to the backbone parameters. It has no I/O, so it is easy to test and to call from any code.

*   **Required fields:** `model_type`, `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `rms_norm_eps`, `num_hidden_layers`, `max_position_embeddings` and `vocab_size`. A missing field raises an error. The converter does not derive `head_dim` from the other sizes, because the Transformers default for Qwen3 is a fixed 128.
*   **Rejected values:** These values raise an error with the field name: `model_type` other than `qwen3`, `hidden_act` other than `silu`, `attention_bias: true`, a non-zero `attention_dropout`, `use_sliding_window: true`, a non-null `rope_scaling`, a non-default `rope_type`, and `layer_types` other than `full_attention`.
*   **Both layouts:** The config must set exactly one of `rope_theta` (Transformers 4) and `rope_parameters` (Transformers 5).
*   **Ignored fields:** Other top-level fields are ignored, e.g. the token ids and `torch_dtype`. `sliding_window` is ignored too, because Transformers reads it only when `use_sliding_window` is set. Unknown keys inside `rope_parameters` raise an error, because each of them changes the RoPE frequencies, e.g. a scaling `factor`.

The converter checks only the options listed above. It does not load weights or the tokenizer, does not choose the dtype and does not compare with arbitrary Hugging Face implementations. `tie_word_embeddings` is accepted: d9d trains the input embeddings and the LM head as separate weights, as it does today.

The `config.json` does not hold the d9d-only fields. The vocabulary split name is a required argument. The pipeline virtual layers keep their defaults.

### Layer 2: the parameters source union

Each family gets a closed union of parameter sources and a factory:

```
d9d/module/model/qwen3_dense/
    huggingface.py      # state mappers + parameters_from_huggingface_qwen3_dense (layer 1)
    params.py           # Qwen3DenseParameters (unchanged)
    params_source.py    # source configs, AnyQwen3DenseParametersSource, build_qwen3_dense_parameters
```

```python
class Qwen3DenseExplicitParametersSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["explicit"] = "explicit"
    parameters: Qwen3DenseParameters


class Qwen3DenseHuggingFaceParametersSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["huggingface"] = "huggingface"
    model_dir: Path
    vocab_split_name: str
    pipeline_num_virtual_layers_pre: int = 0
    pipeline_num_virtual_layers_post: int = 0


AnyQwen3DenseParametersSource = Annotated[
    Qwen3DenseExplicitParametersSource | Qwen3DenseHuggingFaceParametersSource,
    Field(discriminator="kind"),
]


def build_qwen3_dense_parameters(source: AnyQwen3DenseParametersSource) -> Qwen3DenseParameters:
    match source:
        case Qwen3DenseExplicitParametersSource():
            return source.parameters
        case Qwen3DenseHuggingFaceParametersSource():
            config = json.loads((source.model_dir / "config.json").read_text(encoding="utf-8"))
            params = parameters_from_huggingface_qwen3_dense(config, vocab_split_name=source.vocab_split_name)
            return params.model_copy(
                update={
                    "pipeline_num_virtual_layers_pre": source.pipeline_num_virtual_layers_pre,
                    "pipeline_num_virtual_layers_post": source.pipeline_num_virtual_layers_post,
                }
            )
        case _:
            raise ValueError(...)
```

*   **The source configs forbid unknown fields:** A typo such as `pipeline_num_virtual_layers_pr` raises an error instead of being dropped. Only the Hugging Face config itself ignores unknown fields, because it holds metadata that d9d does not use.
*   **The Hugging Face source holds only what the `config.json` lacks:** The d9d-only fields, i.e. the vocabulary split name and the pipeline virtual layers.
*   **The explicit source wraps `Qwen3DenseParameters`:** It does not copy its fields or inherit from it. `Qwen3DenseParameters` gains no discriminator field, because the model, the mappers and the tests use it without any source.
*   **The factory reads the file, not a validator:** The source configs stay pure description and validate without file system access. A local directory is read, with no Hub download and no remote code. Each rank reads the `config.json`, which is a few KiB, so the directory must be available on every node.
*   **Hparam logging:** `model_dump(mode="json")` turns a source into a `ScalarTree` for `ModelProvider.dump_hparams()`. A model provider should log the resolved parameters too, because a path does not record the contents of the `config.json`.

### One converter per family

Each family has its own converter, union and factory, as in DEP-0008. There is no registry and no dispatch on `model_type`.

The user's code already chooses the family: it builds a `Qwen3DenseModel` or a `Qwen3MoEModel`. A per-family factory returns exactly the parameters type of that model. A shared function would return a union of parameter types that the caller must narrow, or would pick the family from the `model_type` in the file. That is behavior decided by the contents of the input, not by an explicit input.

The Qwen3 families share most fields: Qwen3 MoE adds `num_experts` and `experts_top_k`. Shared helpers can be extracted when a second family needs them. For example, the RoPE layout parsing moves next to the RoPE block when Qwen3 MoE needs it. A base class for the shared fields is not needed for that.

### Tests

*   Both sources build the expected parameters. The explicit source does not touch the file system.
*   A missing `kind`, an unknown `kind` and unknown source fields fail validation.
*   The Hugging Face source applies the virtual layer overrides.
*   A missing `config.json`, malformed JSON and each rejected value raise errors with a clear message.
*   Both YAML examples of the Usage section validate through a project config and build parameters.

### Deferred Work

*   **Qwen3 MoE:** The same two layers, in a separate PR. The parameters source does not cover `Qwen3MoEExpertsFormat`. It describes the layout of the checkpoint, not the model, so it stays an argument of the state mappers.
*   **Export:** Writing a Hugging Face config and tokenizer next to the exported weights needs its own design.
*   **RoPE scaling:** The converter rejects a non-null `rope_scaling`, because the Qwen3 models do not support RoPE scaling yet. `RopeScaling` (DEP-0005) is a runtime strategy, not a config. Supporting it needs a serializable scaling config in the parameters, a factory that builds the strategy, and a mapping from the Hugging Face config.

## Usage

The project config takes the union instead of the parameters:

```python
class ProjectConfig(BaseModel):
    model: AnyQwen3DenseParametersSource
    ...

params = build_qwen3_dense_parameters(config.model)
```

Load the architecture from a Hugging Face model:

```yaml
model:
  kind: huggingface
  model_dir: models/Qwen3-0.6B
  vocab_split_name: regular
```

Or write it out explicitly, e.g. for a model trained from scratch:

```yaml
model:
  kind: explicit
  parameters:
    layer:
      hidden_size: 1024
      intermediate_size: 3072
      num_attention_heads: 16
      num_key_value_heads: 8
      rms_norm_eps: 1.0e-06
      head_dim: 128
    num_hidden_layers: 28
    rope_base: 1000000
    max_position_ids: 40960
    split_vocab_size:
      regular: 151936
    split_vocab_order:
      - regular
```

## Backward Compatibility

The library API is fully backward compatible. `Qwen3DenseParameters` does not change, and the union and the factory are new. A project config that declares `model: Qwen3DenseParameters` keeps working with its current YAML.

A project that switches its config to the union must migrate its YAML: add `kind: explicit` and move the fields under `parameters`. The fine-tuning guide moves to the Hugging Face source.

## Alternatives Considered

*   **Only the converter function.** Rejected: each job script reads the `config.json` and calls the function itself. Switching between an explicit and a Hugging Face architecture then needs a code change, not a config change. CONTRIBUTING asks for a discriminated union and a `build_*()` factory when a configurable object has selectable behavior.
*   **A `Qwen3DenseParameters.from_huggingface()` class method.** Rejected: it couples the d9d parameters with the Hugging Face format. The Hugging Face compatibility of a family already lives apart from the model, in `huggingface.py`.
*   **One function for all families**, e.g. `build_parameters_from_huggingface()` with a closed `match` on `model_type`. Rejected: it returns a union of parameter types, and it picks the family from the file instead of from the caller. See "One converter per family".
*   **A converter base class**, e.g. an abstract `HuggingFaceConfigConverter` with one subclass per family. Rejected: the per-family functions need no common interface, because no code calls them polymorphically. Shared parsing goes into plain helper functions.
*   **An open registry keyed by `model_type`.** Rejected: every other union in d9d is closed, and a catalogue model is an in-tree contribution. The Hugging Face state mappers are hard-coded per family for the same reason.
*   **A flat explicit source** (`kind: explicit` next to the parameter fields). Rejected: it needs either a discriminator field on `Qwen3DenseParameters`, which every other user of the class carries without meaning, or a subclass of it, which is inheritance only for the YAML shape.
*   **A `mode="before"` validator on `Qwen3DenseParameters` that also accepts a Hugging Face config.** Rejected: it guesses the format from the contents of the input. The source must be explicit.
*   **`transformers.AutoConfig`.** Rejected: Transformers is only a test dependency of d9d, and its config normalization changes between major versions. The two RoPE layouts come from such a change.
