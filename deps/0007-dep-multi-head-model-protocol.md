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
duplication scales as `3 × N`, and the *same* `3 × N` is mirrored in the `parallelize_*` functions and the HuggingFace
mappers. Each wrapper also hard-binds *exactly one* head, so a model with two heads (multi-task, or a reward head beside
an LM head) cannot be expressed at all.

This proposal replaces the wrappers with one composition primitive: a `DecoderBackbone` protocol, a `TaskHead` contract
confined to head *compute* — parallelization and checkpoint mapping live elsewhere, as separate concerns — and a single
generic `DecoderWithHeads` that composes one backbone with a *mapping* of prebuilt named heads. Heads come from a closed,
discriminated union via a free `build_head` factory, or are passed in directly. All three `3 × N` grids — model wrappers,
`parallelize_*`, and HF mappers — then collapse to `N` backbones plus head-specific work in each layer. A new
architecture writes its backbone once and is usable with any head, or several. Because we are pre-1.0, this
**intentionally breaks** parameter FQNs and the `*For*` class names.

## Motivation

The duplication has **two axes**, a structural ceiling, and it repeats three times over.

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

**Three times over.** The same grid is duplicated in three places: the model wrappers, the `parallelize_*_for_*`
functions (each a per-family backbone routine plus a thin per-head variant that differs only by which head it shards),
and the HF state-dict mappers (a shared backbone mapper plus a one-line per-head rename, per direction). Fixing only the
model classes leaves two-thirds of the duplication standing.

**The ceiling.** Each wrapper is a model *for one task*; several heads on one backbone cannot be expressed. Making the
copies cheaper leaves the grids and the ceiling in place — removing the wrappers is what lifts them.

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

A **head** turns hidden states into named outputs. The module is confined to *compute* — its parallelization and
checkpoint mapping are deliberately kept out of it (separate concerns, handled below). It is an `abc.ABC`:

```python
class TaskHead(nn.Module, ModuleLateInit, abc.ABC):
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

AnyHeadConfig = Annotated[
    CausalLMHeadConfig | ClassificationHeadConfig | EmbeddingHeadConfig, Field(discriminator="kind"),
]
```

A bespoke head a user writes for their own model is a `TaskHead` instance passed directly to `DecoderWithHeads`,
bypassing the union entirely.

### The model

One generic class composes a backbone with a mapping of *prebuilt* heads, generic over the backbone type so
`parallelize_*` keeps the precise `model.model` type. It does not build heads — that is the factory's job — so the class
stays free of the config union:

```python
class DecoderWithHeads(nn.Module, ModuleLateInit, ModuleSupportsPipelining, Generic[TBackbone]):
    def __init__(self, backbone: TBackbone, heads: Mapping[str, TaskHead], stage: PipelineStageInfo): ...
```

The backbone is stored at `self.model` (FQN `model.*`); on the last stage the heads live in an `nn.ModuleDict` (FQN
`heads.<name>.*`). `forward` runs the backbone, then each head over the hidden states, merging each head's outputs under
its mapping key; `infer_stage_outputs` unions the heads' declared shapes; `infer_stage_inputs` delegates to the backbone.

### Multi-head semantics

Supporting more than one head follows from the flat `dict[str, torch.Tensor]` the pipeline passes to the task. That flat
dict — with `"<head>/<key>"` keys *simulating* hierarchy — is what the current pipeline-parallel engine expects;
migrating model outputs to a properly nested structure is deferred to a future change. Given that constraint:

* **Outputs are namespaced by head.** Each head's outputs are merged under its mapping key — `out["lm/logps"]`,
  `out["cls/scores"]`. The key is unique by construction, so two heads of the same type just take different keys
  (`{"cls_a": ..., "cls_b": ...}` → `out["cls_a/scores"]`, `out["cls_b/scores"]`) with no extra config and no collision
  check. Backbone outputs (`hidden_states`) stay top-level.
* **Loss combination is the task's job.** The model only emits per-head tensors; `BaseTask.compute_loss` reads the keys
  it wants and combines them. The model holds no loss policy.
* **No "primary" output.** The pipeline contract is the whole dict; the task selects keys.
* **Head inputs are routed, not enumerated.** Backbone inputs stay typed kwargs; head-specific tensors (`labels`,
  `pooling_mask`) arrive via `**head_inputs`, and each head reads what it needs. An open set of heads cannot be named
  kwargs — this mapping is the irreducible cost of genuine multi-head, and it documents each input on its consumer.

### Parallelization & checkpoint mapping

`parallelize_*_for_*` and the HF mappers carry the same `3 × N`. Both are concerns *separate* from compute, so both move into their own layer — each a single function over a head, beside the unchanged per-family
backbone routine:

```python
# d9d/module/parallelism/...
def parallelize_task_head(head: TaskHead, dist_context: DistributedContext) -> None: ...
# d9d/module/model/.../huggingface.py
def task_head_hf_mapper(head: TaskHead, prefix: str) -> ModelStateMapper: ...
```

They differ only in how much they vary. Head **sharding is uniform** — every head is HSDP on the dense mesh, so
`parallelize_task_head` is one implementation for all heads (a head needing different sharding is the point to branch —
not before). The HF **rename is per-head-type** (`score.weight ↔ heads.cls.score.weight`, the vocab-aware LM rename, …)
but family-independent, so `task_head_hf_mapper` has one arm per head type. The composed model applies the per-family
backbone routine, then iterates `heads` through these two. Both grids drop to `N` backbone routines plus per-head
work — shared across heads for sharding, per-type for mapping.

## Usage

A provider builds the backbone, then the heads (via the factory, or directly), then composes them:

```python
backbone = Qwen3MoEModel(cfg.model, stage, hidden_states_snapshot_mode=..., enable_checkpointing=...)
heads = {"lm": build_head(CausalLMHeadConfig(), backbone=backbone, stage=stage)}
model = DecoderWithHeads(backbone, heads, stage)
```

A second head is a one-line change; the task then reads both keys:

```python
heads = {
    "lm": build_head(CausalLMHeadConfig(), backbone=backbone, stage=stage),
    "cls": build_head(ClassificationHeadConfig(num_labels=3), backbone=backbone, stage=stage),
}
# loss = lm_loss(out["lm/logps"]) + alpha * cls_loss(out["cls/scores"])
```

## Backward Compatibility

A deliberate, pre-1.0 break; every cost is one-time and mechanical:

- **Classes & functions.** The `*For*` models and their `*For*Parameters` are removed, along with the per-head
  `parallelize_*_for_*` and `mapper_*_for_*` functions; providers compose `DecoderWithHeads`, and head sharding and
  renames move to `parallelize_task_head` / `task_head_hf_mapper`. Backbone params and the per-family backbone
  `parallelize`/`mapper` routines are unchanged.
- **Parameter FQNs.** Heads move `lm_head.* → heads.lm.*` (etc.); the backbone stays `model.*`. Existing checkpoints
  need a one-pass key remap; the HuggingFace renames update accordingly (now emitted by the per-head-type mapper).
- **Output keys.** Head outputs are now namespaced (`out["lm/logps"]` instead of `out["logps"]`); code reading pipeline
  results updates to the namespaced keys.
- **Tests.** The suites under `test/d9d_test/modules/model/sequence/` (HF parity + state-dict round-trip across both
  families × all heads) update to the new construction and are the regression gate.

## Alternatives Considered

**Per-family thin wrappers (the earlier draft of this DEP).** Keep one wrapper per task per family, factoring shared glue
into three generic base classes each subclassed in ~6 lines. *Rejected:* cheaper copies, but the `3 × N` grid remains
(in all three axes) and a multi-head model still cannot be expressed. Its byte-stable-FQN requirement is what forced the
per-family subclass; lifting it (pre-1.0) is what unlocks composition.

**A class that builds its own heads (`from_configs`).** Give `DecoderWithHeads` a classmethod that constructs heads from
a config mapping. *Rejected:* it couples the composition class to the head-config union; keeping construction in the free
`build_head` factory leaves the class a pure composition of prebuilt modules and lets custom heads in on equal footing.

**Dynamic class factory.** Generating each wrapper at import time. *Rejected:* metaprogramming defeats `ty`, erases
docstrings and IDE navigation, and removes no duplication that composition does not.
