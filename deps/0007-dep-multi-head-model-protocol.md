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

**Across heads.** In `qwen3_moe/model.py` the three wrappers are ~70-85 lines each and ~90% identical. The only
per-head differences are:

| Concern | CausalLM | Classification | Embedding |
|---|---|---|---|
| head submodule (attr) | `lm_head` | `cls_head` | `embedding_head` |
| `SharedInput` type | `SequenceCausalLMShared` | `SequencePoolingShared` | `SequencePoolingShared` |
| extra shared field | `labels` | `pooling_mask` | `pooling_mask` |
| `PipelineOutput` type | `SequenceCausalLMOutput` | `SequenceClassificationOutput` | `SequenceEmbeddingOutput` |

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

A **backbone** maps stage inputs to hidden states, reports its `hidden_size` and split-vocabulary layout, and
supports late init and pipelining — exactly the surface the wrappers already call on `self.model`. It is the sequence
backbone the pipelining PyTree IO of [DEP-0010](0010-dep-pipelining-pytree-io.md) already describes, with the shared
dimensions the heads derive from made public:

```python
@typing.runtime_checkable
class DecoderBackbone(
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequenceShared, SequenceTransfer[torch.Tensor]
    ],
    Protocol,
):
    hidden_size: int
    split_vocab_size: dict[str, int]
    split_vocab_order: list[str]
```

`Qwen3MoEModel` and `Qwen3DenseModel` conform by exposing those three dimensions (a three-line addition; all already
arrive via params).

A **head** turns hidden states into its own typed output PyTree. The module is confined to *compute* — its
parallelization and checkpoint mapping are kept out of it (separate concerns, below). Alongside the shared
`hidden_states` it reads its **own** shared input, so `labels`, `pooling_mask` and any future signal are typed rather
than looked up by string:

```python
class TaskHead(nn.Module, ModuleLateInit, abc.ABC, Generic[THeadShared, THeadOutput]):
    @abc.abstractmethod
    def forward(self, hidden_states: torch.Tensor, shared: THeadShared) -> THeadOutput: ...
```

The three existing heads in `d9d/module/block/head/` implement this directly — `SplitLanguageModellingHead` as
`TaskHead[SequenceCausalLMHeadShared, SequenceCausalLMOutput]`, and so on. A custom head just meets this small contract;
its sharding and state mapping are then supplied at composition, exactly as for built-in heads. Nothing else is
required: the engine sizes its buffers from `stage_transfer_spec`, which describes only what crosses a stage boundary,
and the last stage's outgoing edge never transfers — so a head declares no shapes.

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
class DecoderWithHeads(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[SequenceInput, SequenceTransfer[torch.Tensor], SequenceHeadsShared, SequenceHeadsOutput],
    Generic[TBackbone],
):
    def __init__(self, backbone: TBackbone, heads: Mapping[str, TaskHead], stage: PipelineStageInfo):
        self.model = backbone                                   # FQN: model.*
        self._stage = stage
        if stage.is_current_stage_last:
            self.heads = nn.ModuleDict(...)                     # FQN: heads.<name>.*

    def forward(self, inputs, shared: SequenceHeadsShared):
        model_outputs = self.model(inputs, shared.sequence)
        if not self._stage.is_current_stage_last:
            return model_outputs
        return {name: head(model_outputs.hidden_states, shared.heads[name]) for name, head in self.heads.items()}
```

The `SharedInput` carries both halves of the composition — `shared.sequence` for the backbone, `shared.heads[name]` for
each head — so routing is by *name*, not by string keys in a flat bag:

```python
@dataclasses.dataclass
class SequenceHeadsShared:
    sequence: SequenceShared
    heads: Mapping[str, Any]
```

`reset_parameters` delegates to the backbone and the heads; `stage_transfer_spec` delegates to the backbone, since the
heads only run on the last stage, whose outgoing edge never transfers. The backbone (`self.model`) and the heads
(`self.heads`) are public, so a provider parallelizes each independently.

### Multi-head semantics

The pipeline moves arbitrary PyTrees between stages and to the task, so more than one head needs no encoding trick —
a mapping keyed by head name *is* the `PipelineOutput`:

* **Outputs are keyed by head.** `out["lm"]` is a `SequenceCausalLMOutput`, `out["cls"]` a
  `SequenceClassificationOutput` — each head's own typed output, unflattened. The key is unique by construction, so two
  heads of the same type just take different keys (`{"cls_a": ..., "cls_b": ...}`) with no extra config and no collision
  check.
* **Loss combination is the task's job.** The model only emits per-head outputs; `BaseTask.compute_loss` reads the ones
  it wants and combines them (`ctx.pipeline_results["lm"].logps`). The model holds no loss policy.
* **No "primary" output.** The pipeline contract is the whole mapping; the task selects heads.
* **Inputs are routed by name, not enumerated.** Each head's shared input is a typed PyTree the head declares
  (`SequenceCausalLMHeadShared`, `SequencePoolingHeadShared`, …), delivered under the head's own key. An open set of
  heads cannot be named kwargs, so the mapping level is irreducible — but everything below it stays typed, and a new
  input to a head is a field on that head's shared dataclass, not a new signature anywhere else.

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
# shared = SequenceHeadsShared(
#     sequence=SequenceShared(position_ids=...),
#     heads={"lm": SequenceCausalLMHeadShared(labels=...),
#            "cls": SequencePoolingHeadShared(pooling_mask=...)},
# )
# loss = lm_loss(out["lm"].logps) + alpha * cls_loss(out["cls"].scores)
```

## Backward Compatibility

A deliberate, pre-1.0 break; every cost is one-time and mechanical:

- **Classes & functions.** The `*For*` models and their `*For*Parameters` are removed, along with the per-head
  `parallelize_*_for_*` and `mapper_*_for_*` functions; providers compose `DecoderWithHeads`, head sharding moves to
  `parallelize_task_head`, and head renames move to one mapper per head type. The backbone exposes `hidden_size` and
  its split-vocab layout publicly; backbone params, its `forward`, and the per-family backbone `parallelize`/`mapper`
  routines are otherwise unchanged.
- **Parameter FQNs.** Heads move `lm_head.* → heads.lm.*` (etc.); the backbone stays `model.*`. Existing checkpoints
  need a one-pass key remap; the HuggingFace renames update accordingly (now emitted by the per-head-type mapper).
- **IO types.** `SequenceCausalLMShared` / `SequencePoolingShared` (a backbone shared input bundled with one head's
  payload) are replaced by `SequenceHeadsShared` plus the per-head `SequenceCausalLMHeadShared` /
  `SequencePoolingHeadShared`. The `PipelineOutput` becomes a mapping keyed by head name, so `results.logps` becomes
  `results["lm"].logps`; the per-head output dataclasses themselves are unchanged.
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

**A flat `dict[str, torch.Tensor]` with `"<head>/<key>"` keys.** Keep the model's IO a single flat bag and simulate
hierarchy in the key strings, with a `HeadInputs: Mapping[str, torch.Tensor | None]` for the inputs and a matching
`infer_output_shapes` on every head. *Rejected:* this was the shape of an earlier draft, written when the engine could
only move flat dicts. [DEP-0009](0009-dep-dataflow-refactoring.md) and
[DEP-0010](0010-dep-pipelining-pytree-io.md) removed that constraint — the engine now carries arbitrary PyTrees and
sizes buffers from `stage_transfer_spec` alone — so the string namespacing and the shape-inference method would both be
pure ceremony, and every head input and output would lose its type.

**A single flat output dataclass with per-head fields.** *Rejected:* the head set is open and named at composition, so
the fields cannot be known in advance; the mapping keyed by head name is the minimum structure that supports it.

**Dynamic class factory.** Generating each wrapper at import time. *Rejected:* metaprogramming defeats `ty`, erases
docstrings and IDE navigation, and removes no duplication that composition does not.
