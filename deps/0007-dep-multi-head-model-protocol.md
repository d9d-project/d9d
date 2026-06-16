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

Every architecture in `d9d` ships three near-identical task wrappers — `*ForCausalLM`, `*ForClassification`,
`*ForEmbedding` — that share a backbone and differ only in their head. They are copy-pasted across families, so the
duplication scales as `3 × N`. Each wrapper also hard-binds *exactly one* head, so a model with two heads (multi-task,
or a reward head beside an LM head) cannot be expressed at all.

This proposal replaces the wrappers with one composition primitive: a `DecoderBackbone` protocol, a `TaskHead` contract,
and a single generic `DecoderModel` that composes one backbone with a *mapping* of named heads built from a closed,
discriminated union of head configs. A new architecture writes its backbone once and is usable with any head, or several.
Because we are pre-1.0, this **intentionally breaks** parameter FQNs and the `*For*` class names.

## Motivation

The duplication has **two axes**, plus a structural ceiling.

**Across heads.** In `qwen3_moe/model.py` the three wrappers are ~110 lines each and ~90% identical. The only per-head
differences are:

| Concern | CausalLM | Classification | Embedding |
|---|---|---|---|
| head submodule (attr) | `lm_head` | `cls_head` | `embedding_head` |
| extra `forward` input | `labels` | `pooling_mask` | `pooling_mask` |
| output key | `logps` | `scores` | `embeddings` |
| inferred output shape | `input_ids.shape` | `(B, num_labels)` | `(B, embedding_dim)` |

Everything else is mechanical glue.

**Across families.** `qwen3_dense/model.py` repeats all three verbatim, and any change to the wrapper contract must be
applied `3 × N` times.

**The ceiling.** Each wrapper is a model *for one task*; several heads on one backbone cannot be expressed. Making the
copies cheaper leaves both the grid and the ceiling in place — removing the wrappers is what lifts them.

## Design Proposal

### Contracts

A **backbone** maps stage inputs to hidden states, reports its `hidden_size`, and supports late init and pipelining —
exactly the surface the wrappers already call on `self.model`:

```python
@typing.runtime_checkable
class DecoderBackbone(ModuleLateInit, ModuleSupportsPipelining, Protocol):
    hidden_size: int

    def forward(self, input_ids=None, hidden_states=None, position_ids=None,
                hidden_states_snapshot=None, hidden_states_agg_mask=None) -> dict[str, torch.Tensor | None]: ...
```

`Qwen3MoEModel` and `Qwen3DenseModel` conform by exposing `hidden_size` (a one-line addition; both already receive it
via params).

A **head** turns hidden states into named outputs. It is an `abc.ABC` and declares the keys it writes, so collisions are caught at build time:

```python
class TaskHead(nn.Module, ModuleLateInit, abc.ABC):
    output_keys: frozenset[str]

    @abc.abstractmethod
    def forward(self, hidden_states: torch.Tensor, inputs: Mapping[str, torch.Tensor | None]) -> dict[str, torch.Tensor]: ...

    @abc.abstractmethod
    def infer_output_shapes(self, pipeline_inputs: dict[str, torch.Tensor], n_microbatches: int) -> dict[str, torch.Tensor]: ...
```

The three existing heads in `d9d/module/block/head/` implement this directly — no parallel hierarchy.

### Head configuration

Heads are selected by a closed, discriminated union resolved by a free `build_head(config, *, backbone, stage) ->
TaskHead` with an exhaustive `match` — the `AnyDecayGateParameters` / `_build_decay_gate` pattern. A config carries only
*task-specific* fields; backbone-shared dimensions (`hidden_size`, the LM split-vocab layout) are derived from the
backbone, so configs stay small and nothing is specified twice.

```python
class ClassificationHeadConfig(BaseModel):
    kind: Literal["classification"] = "classification"
    num_labels: int
    dropout: float
    output_key: str = "scores"

AnyHeadConfig = Annotated[
    CausalLMHeadConfig | ClassificationHeadConfig | EmbeddingHeadConfig, Field(discriminator="kind"),
]
```

A bespoke head a user writes for their own model is a `TaskHead` instance passed directly to `DecoderModel`, bypassing
the union entirely.

### The model

One generic class composes a backbone with a mapping of named heads, generic over the backbone type so `parallelize_*`
keeps the precise `model.model` type:

```python
class DecoderModel(nn.Module, ModuleLateInit, ModuleSupportsPipelining, Generic[TBackbone]):
    def __init__(self, backbone: TBackbone, heads: Mapping[str, TaskHead], stage: PipelineStageInfo): ...

    @classmethod
    def from_configs(cls, backbone: TBackbone, head_configs: Mapping[str, AnyHeadConfig],
                     stage: PipelineStageInfo) -> "DecoderModel[TBackbone]": ...
```

The backbone is stored at `self.model` (FQN `model.*`); on the last stage the heads live in an `nn.ModuleDict` (FQN
`heads.<name>.*`). `forward` runs the backbone, then each head over the hidden states and merges their outputs;
`infer_stage_outputs` unions the heads' declared shapes; `infer_stage_inputs` delegates to the backbone. `from_configs`
is the common path; the bare constructor is the seam for pre-built or custom heads.

### Multi-head semantics

Supporting more than one head follows from the flat `dict[str, torch.Tensor]` the pipeline already passes to the task:

* **Output keys are flat and disjoint.** Each head declares `output_keys`; construction rejects any overlap (between
  heads, or with `hidden_states`) with a `ValueError`, instead of silently overwriting. Two heads of the same type stay
  distinct via each config's `output_key`. The FQN namespace (`heads.<name>.*`) is independent of the output-key one.
* **Loss combination is the task's job.** The model only emits per-head tensors; `BaseTask.compute_loss` — which already
  reads `pipeline_results["logps"]` — reads the keys it wants and combines them. The model holds no loss policy.
* **No "primary" output.** The pipeline contract is the whole dict; the task selects keys.
* **Head inputs are routed, not enumerated.** Backbone inputs stay typed kwargs; head-specific tensors (`labels`,
  `pooling_mask`) arrive via `**head_inputs`, and each head reads what it needs. An open set of heads cannot be named
  kwargs — this mapping is the irreducible cost of genuine multi-head, and it documents each input on its consumer.

## Usage

A provider builds the backbone, then composes heads from config:

```python
backbone = Qwen3MoEModel(cfg.model, stage, hidden_states_snapshot_mode=..., enable_checkpointing=...)
model = DecoderModel.from_configs(backbone, head_configs={"lm": CausalLMHeadConfig()}, stage=stage)
```

A second head is a one-line change; the task then reads both keys:

```python
head_configs={"lm": CausalLMHeadConfig(), "cls": ClassificationHeadConfig(num_labels=3)}
# loss = lm_loss(out["logps"]) + alpha * cls_loss(out["scores"])
```

## Backward Compatibility

A deliberate, pre-1.0 break; every cost is one-time and mechanical:

- **Classes.** The `*For*` models and their `*For*Parameters` are removed; providers call `DecoderModel.from_configs`.
  Backbone params are unchanged; head-specific fields move onto the head configs.
- **Parameter FQNs.** Heads move `lm_head.* → heads.lm.*` (etc.); the backbone stays `model.*`. Existing checkpoints
  need a one-pass key remap; the HuggingFace renames in `huggingface.py` update accordingly.
- **`parallelize_*`.** Reaches `model.heads["lm"]` instead of `model.lm_head`; `model.model` is unchanged.
- **Tests.** The suites under `test/d9d_test/modules/model/sequence/` (HF parity + state-dict round-trip across both
  families × all heads) update to the new construction and are the regression gate.

## Alternatives Considered

**Per-family thin wrappers (the earlier draft of this DEP).** Keep one wrapper per task per family, factoring shared glue
into three generic base classes each subclassed in ~6 lines. *Rejected:* cheaper copies, but the `3 × N` grid remains and
a multi-head model still cannot be expressed. Its byte-stable-FQN requirement is what forced the per-family subclass;
lifting it (pre-1.0) is what unlocks composition.

**Single wrapper + injected head, FQNs preserved.** The adopted design, but constrained to keep `lm_head` / `cls_head`
names — forcing dynamic `add_module` registration and untyped `**kwargs`. *Rejected in that form:* with a sanctioned FQN
change an `nn.ModuleDict` gives clean static names and generalizes to many heads.

**Dynamic class factory.** Generating each wrapper at import time. *Rejected:* metaprogramming defeats `ty`, erases
docstrings and IDE navigation, and removes no duplication that composition does not.
