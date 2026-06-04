---
DEP: 0007
Title: Multi-Head Model Protocol
Author: Daniil Sergeev @DaniilSergeev17
Status: Draft
Type: Feature
Created: 2026-06-04
---

# DEP-0007: Multi-Head Model Protocol

## Abstract

Every model architecture in `d9d` currently ships three near-identical task wrappers — `*ForCausalLM`, `*ForClassification`, `*ForEmbedding` — that share a backbone and differ only in their head. The wrappers are copy-pasted across architectures, so the duplication scales as `3 × N` for `N` families.

This proposal introduces a **multi-head model protocol**: a `DecoderBackbone` structural protocol describing the contract a decoder backbone already fulfils, plus three generic, backbone-agnostic task models (`CausalLMModel`, `ClassificationModel`, `EmbeddingModel`) that compose any conforming backbone with a head. The task models are written *once* under `d9d/module/model/task/`. Each concrete model (`Qwen3MoEForCausalLM`, …) becomes a thin subclass that constructs its backbone and forwards head configuration.

Existing backbones (`Qwen3MoEModel`, `Qwen3DenseModel`) satisfy the protocol structurally with zero changes. Public class names, constructor signatures, and — critically — parameter fully-qualified names stay byte-stable, so HuggingFace mappers, `parallelize_*` functions, checkpoints, and tests are unaffected. Adding a new architecture drops from three hand-written wrappers (~330 lines) to three ~6-line subclasses.

## Motivation

The duplication has **two axes**.

**Across heads.** Within `d9d/module/model/qwen3_moe/model.py`, `Qwen3MoEForCausalLM`, `Qwen3MoEForClassification`, and `Qwen3MoEForEmbedding` are ~110 lines each and ~90% identical. The only per-head differences are:

| Concern | CausalLM | Classification | Embedding |
|---|---|---|---|
| head submodule (attr) | `lm_head` | `cls_head` | `embedding_head` |
| extra `forward` kwarg | `labels` | `pooling_mask` | `pooling_mask` |
| output key | `logps` | `scores` | `embeddings` |
| inferred output shape | `input_ids.shape` | `(B, num_labels)` | `(B, embedding_dim)` |

Everything else — `self.model = …Model(...)`, the backbone forward call, `reset_parameters` delegation, and the byte-for-byte identical `infer_stage_inputs_from_pipeline_inputs` — is mechanical glue unrelated to the head.

**Across families.** `d9d/module/model/qwen3_dense/model.py` repeats the same three wrappers verbatim. Every new architecture re-copies all three, and any change to the wrapper contract (a new pipelining method, a new head kwarg) must be applied `3 × N` times.

## Design Proposal

### The `DecoderBackbone` protocol

A backbone is any module that maps stage inputs to `{"hidden_states", "hidden_states_snapshot"}` and supports late init and pipelining metadata. This is exactly the surface the task wrappers already call on `self.model`, so the protocol composes the two existing protocols plus the forward contract:

```python
@typing.runtime_checkable
class DecoderBackbone(ModuleLateInit, ModuleSupportsPipelining, Protocol):
    """A pipelinable decoder backbone emitting hidden states for a task head."""

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        hidden_states: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        hidden_states_snapshot: torch.Tensor | None = None,
        hidden_states_agg_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]: ...
```

`Qwen3MoEModel` and `Qwen3DenseModel` already conform — no edits to either backbone.

### Generic task models

Three classes, written once, generic over the concrete backbone type so that downstream code (e.g. `parallelize_*`) keeps the precise `model.model` type:

```python
TBackbone = TypeVar("TBackbone", bound=DecoderBackbone)

class CausalLMModel(nn.Module, ModuleLateInit, ModuleSupportsPipelining, Generic[TBackbone]):
    """Composes a decoder backbone with a split language-modelling head."""

    def __init__(
        self,
        backbone: TBackbone,
        stage: PipelineStageInfo,
        *,
        split_vocab_size: dict[str, int],
        split_order: Sequence[str],
        hidden_size: int,
    ):
        super().__init__()
        self.model: TBackbone = backbone

    def forward(self, input_ids=None, hidden_states=None, position_ids=None,
                hidden_states_snapshot=None, hidden_states_agg_mask=None, labels=None):
        ...

    def reset_parameters(self):
        ...

    def infer_stage_inputs_from_pipeline_inputs(self, inputs, n_microbatches):
        ...

    def infer_stage_outputs_from_pipeline_inputs(self, inputs, n_microbatches):
        ...
```

`ClassificationModel` and `EmbeddingModel` follow the same shape, differing only in the four cells of the table above (head class + attr, the `pooling_mask` kwarg, the output key, and the inferred shape). The `infer_stage_inputs_from_pipeline_inputs` delegation and the backbone storage/forward/reset wiring are identical across all three.

### Concrete models become thin subclasses

```python
# d9d/module/model/qwen3_moe/model.py

class Qwen3MoEForCausalLM(CausalLMModel[Qwen3MoEModel]):
    def __init__(self, params, stage, hidden_states_snapshot_mode, enable_checkpointing):
        backbone = Qwen3MoEModel(params.model, stage, hidden_states_snapshot_mode, enable_checkpointing)
        super().__init__(
            backbone, stage,
            split_vocab_size=params.model.split_vocab_size,
            split_order=params.model.split_vocab_order,
            hidden_size=params.model.layer.hidden_size,
        )
```

The backbone classes (`Qwen3MoEModel`, `Qwen3DenseModel`) are kept verbatim. The three wrappers per family collapse to three subclasses of this form.

## Usage

A new architecture writes its backbone once, then derives the three task models — no per-head pipelining or inference logic:

```python
from d9d.module.model.task import CausalLMModel, ClassificationModel, EmbeddingModel


class MyModel(nn.Module, ModuleLateInit, ModuleSupportsPipelining):
    # ... emits {"hidden_states", "hidden_states_snapshot"}; conforms to DecoderBackbone structurally
    ...


class MyModelForCausalLM(CausalLMModel[MyModel]):
    def __init__(self, params, stage, hidden_states_snapshot_mode, enable_checkpointing):
        backbone = MyModel(params.model, stage, hidden_states_snapshot_mode, enable_checkpointing)
        super().__init__(
            backbone, stage,
            split_vocab_size=params.model.split_vocab_size,
            split_order=params.model.split_vocab_order,
            hidden_size=params.model.layer.hidden_size,
        )


class MyModelForClassification(ClassificationModel[MyModel]):
    def __init__(self, params, stage, hidden_states_snapshot_mode, enable_checkpointing):
        backbone = MyModel(params.model, stage, hidden_states_snapshot_mode, enable_checkpointing)
        super().__init__(
            backbone, stage,
            num_labels=params.num_labels,
            dropout=params.classifier_dropout,
            hidden_size=params.model.layer.hidden_size,
        )
```

End-user training scripts are unchanged — they keep constructing `Qwen3MoEForCausalLM(params=..., stage=..., hidden_states_snapshot_mode=..., enable_checkpointing=...)` exactly as before.

## Backward Compatibility

**Fully backward compatible.** The migration preserves every observable contract:

- **Class names and constructor signatures** are unchanged; `Qwen3MoEForCausalLM` etc. remain importable and subclassing keeps `isinstance` semantics.
- **Parameter FQNs** stay `model.*`, `lm_head.*`, `cls_head.*`, `embedding_head.*`. The head submodule attribute names are written literally in the generic classes, so existing checkpoints load, and the HuggingFace mappers in `huggingface.py` (e.g. `ModelStateMapperRename("score.weight", "cls_head.score.weight")`) need no changes.
- **`parallelize_*` functions** keep reaching into `model.model`, `model.lm_head`, … — and because the task models are `Generic[TBackbone]`, `model.model` retains its concrete type for the type checker.
- **`forward` signatures** keep their typed, documented per-head kwargs (`labels` / `pooling_mask`).

No user migration is required.

## Alternatives Considered

### Single wrapper + injected head-strategy object

A single `DecoderForTask` parameterized by a `HeadAdapter` object encapsulating head construction, forward invocation, and output inference. **Rejected.** It collapses the three typed `forward`s into `**kwargs` (losing `labels` / `pooling_mask` from the signature and docstrings), and must register the head under a dynamic attribute name via `add_module` to preserve FQNs. More indirection, weaker typing, and at odds with the codebase's typed-signature conventions — for no extra leverage over three explicit classes.

### Dynamic class factory

A `make_causal_lm_class(backbone_cls, params_cls) -> type` generating each wrapper at import time. **Rejected.** Metaprogramming defeats `ty`, erases docstrings and IDE navigation, and obscures the public classes that users import and `isinstance`-check.

### Plain helper extraction

Keep the three hand-written wrappers per family but pull the identical `infer_stage_inputs` delegation into a shared function. **Rejected.** It only dents the across-heads axis; a new architecture still hand-writes three full wrappers, leaving the across-families duplication — the dominant cost — untouched.
