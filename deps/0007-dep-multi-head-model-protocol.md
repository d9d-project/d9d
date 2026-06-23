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
exactly the surface the wrappers already call on `self.model`. Its `forward` keeps the typed standard inputs plus a
`**inputs` catch-all, so a backbone may consume additional tensors without a contract change:

```python
HeadInputs: TypeAlias = Mapping[str, torch.Tensor | None]


@typing.runtime_checkable
class DecoderBackbone(ModuleLateInit, ModuleSupportsPipelining, Protocol):
    hidden_size: int

    def forward(self, input_ids=None, hidden_states=None, position_ids=None,
                hidden_states_snapshot=None, hidden_states_agg_mask=None, **inputs) -> dict[str, torch.Tensor | None]: ...
```

`Qwen3MoEModel` and `Qwen3DenseModel` conform by exposing `hidden_size` (a one-line addition; both already receive it
via params).

A **head** turns hidden states into named outputs. The module is confined to *compute* — its parallelization and
checkpoint mapping are kept out of it (separate concerns, below). Alongside the shared `hidden_states`, it reads any
head-specific tensors (`labels`, `pooling_mask`, targets, …) from a `HeadInputs` mapping, so a new input needs no
signature change:

```python
class TaskHead(nn.Module, ModuleLateInit, abc.ABC):
    @abc.abstractmethod
    def forward(self, hidden_states: torch.Tensor, inputs: HeadInputs) -> dict[str, torch.Tensor]: ...

    @abc.abstractmethod
    def infer_output_shapes(self, pipeline_inputs: dict[str, torch.Tensor], n_microbatches: int) -> dict[str, torch.Tensor]: ...
```

The three existing heads in `d9d/module/block/head/` implement this directly. A custom head — even a plain `nn.Module` —
just meets this small contract (subclass `TaskHead`, or wrap the module in a thin adapter adding `infer_output_shapes`);
its sharding and state mapping are then supplied at composition, exactly as for built-in heads.

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

One generic class composes a backbone with a mapping of *prebuilt* heads, generic over the backbone type so the backbone
keeps its precise type for `parallelize_*`. It does not build heads — that is the factory's job — so the class stays free
of the config union:

```python
class DecoderWithHeads(nn.Module, ModuleLateInit, ModuleSupportsPipelining, Generic[TBackbone]):
    def __init__(self, backbone: TBackbone, heads: Mapping[str, TaskHead], stage: PipelineStageInfo):
        self.model = backbone                                   # FQN: model.*
        self._stage = stage
        if stage.is_current_stage_last:
            self.heads = nn.ModuleDict(...)                     # FQN: heads.<name>.*

    def forward(self, input_ids=None, hidden_states=None, position_ids=None,
                hidden_states_snapshot=None, hidden_states_agg_mask=None, **inputs):
        out = self.model(input_ids=input_ids, hidden_states=hidden_states, position_ids=position_ids,
                         hidden_states_snapshot=hidden_states_snapshot,
                         hidden_states_agg_mask=hidden_states_agg_mask, **inputs)
        if self._stage.is_current_stage_last:
            for name, head in self.heads.items():
                out.update({f"{name}/{k}": v for k, v in head(out["hidden_states"], inputs).items()})
        return out
```

The pipeline engine invokes the model as `module(**stage_inputs)` (the flat-dict contract below): the typed standard
inputs flow to the backbone, and the `**inputs` catch-all is both forwarded to the backbone (it absorbs extras via its
own `**inputs`) and handed to each head as its `HeadInputs`. `reset_parameters` and `infer_stage_outputs` delegate to the
backbone and the heads; `infer_stage_inputs` delegates to the backbone. The backbone (`self.model`) and the heads
(`self.heads`) are public, so a provider parallelizes each independently.

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
* **Inputs are routed, not enumerated.** The standard inputs stay typed kwargs; everything else (`labels`,
  `pooling_mask`, future targets, …) arrives via `**inputs` and reaches whatever consumes it — the backbone (via its
  `**inputs`) or a head (via its `HeadInputs`). An open set of heads cannot be named kwargs; this mapping is the
  irreducible cost of genuine multi-head, and it makes a new input a no-signature-change addition.

### Parallelization & checkpoint mapping

`parallelize_*_for_*` and the HF mappers carry the same `3 × N`. Both are concerns *separate* from compute, so both stay
out of the `TaskHead` module, in their own layer — and they are not symmetric:

* **Sharding is uniform** — every head is HSDP on the dense mesh, so one function covers all heads (a head needing
  different sharding is the point to branch — not before):

  ```python
  # d9d/module/parallelism/...
  def parallelize_task_head(head: TaskHead, dist_context: DistributedContext) -> None: ...
  ```

* **Mapping is per-head-type** — each head's HF rename genuinely differs (`score.weight ↔ heads.cls.score.weight`, the
  vocab-aware LM rename, …) but is family-independent, so it is one standalone mapper *per head type*:

  ```python
  # d9d/module/model/.../huggingface.py
  def hf_mapper_for_lm_head(head, prefix) -> ModelStateMapper: ...
  def hf_mapper_for_cls_head(head, prefix) -> ModelStateMapper: ...
  def hf_mapper_for_embedding_head(head, prefix) -> ModelStateMapper: ...
  ```

Because `DecoderWithHeads` exposes the backbone (`self.model`) and the heads (`self.heads`), a provider drives each
independently — the per-family backbone routine on the backbone, then a loop over the heads:

```python
def parallelize_model_stage(self, ctx):
    parallelize_qwen3_moe_model(ctx.dist_context, ctx.model.model, ctx.stage)   # backbone
    if ctx.stage.is_current_stage_last:
        for head in ctx.model.heads.values():
            parallelize_task_head(head, ctx.dist_context)                       # heads
```

Both grids drop to `N` backbone routines plus per-head work — one shared sharding function, one mapper per head type.

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
  `parallelize_*_for_*` and `mapper_*_for_*` functions; providers compose `DecoderWithHeads`, head sharding moves to
  `parallelize_task_head`, and head renames move to one mapper per head type. The backbone `forward` gains `**inputs`;
  backbone params and the per-family backbone `parallelize`/`mapper` routines are otherwise unchanged.
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

**A typed input DTO instead of the `HeadInputs` mapping.** Replace the mapping with a frozen dataclass of named input
fields. *Rejected:* the input set is open (any head, any future signal) and the pipeline engine calls the model with a
flat `**dict`, so a DTO would still need an `extra: dict` escape hatch plus a dict→DTO conversion at the boundary. A
named `Mapping` alias keeps the standard inputs typed where it matters and stays aligned with the engine's flat-dict
contract.

**Dynamic class factory.** Generating each wrapper at import time. *Rejected:* metaprogramming defeats `ty`, erases
docstrings and IDE navigation, and removes no duplication that composition does not.
