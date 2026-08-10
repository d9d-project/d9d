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

This proposal replaces the wrappers with head-generic composition: a `DecoderBackbone` protocol, a `TaskHead` contract
confined to head *compute* — parallelization and checkpoint mapping live elsewhere, as separate concerns — and two
generic composition classes, `DecoderWithHead` for one head and `DecoderWithHeads` for a *mapping* of named heads, with
one thin subclass per built-in head over the former. Heads come from a closed, discriminated union via a free
`build_decoder_head` factory, or are passed in directly. The `3 × N` model-wrapper grid collapses to `N` backbones plus
those head-generic classes; `parallelize_*` and the HF mappers keep one entry *per head type*, so each stays free to
diverge. A new architecture writes its backbone once and is usable with any head, or several. Because we are pre-1.0,
this **intentionally breaks** parameter FQNs and the `*For*` class names.

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
dimensions the heads derive from exposed as **read-only properties** — they are fixed at construction, and the protocol
says so:

```python
@typing.runtime_checkable
class DecoderBackbone(
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequenceShared, SequenceTransfer[torch.Tensor]
    ],
    Protocol,
):
    @property
    def hidden_size(self) -> int: ...
    @property
    def split_vocab_size(self) -> Mapping[str, int]: ...
    @property
    def split_vocab_order(self) -> Sequence[str]: ...
```

`Qwen3MoEModel` and `Qwen3DenseModel` conform by exposing those three dimensions (all already arrive via params).

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
`TaskHead[SequenceCausalLMHeadShared, SequenceCausalLMOutput]`, and so on. Their IO dataclasses live beside them in
`d9d/module/block/head/io.py`, so the head package depends on nothing under `d9d/module/model/` and the composition
layer above it is free to import heads. A custom head just meets this small contract;
its sharding and state mapping are then supplied at composition, exactly as for built-in heads. Nothing else is
required: the engine sizes its buffers from `stage_transfer_spec`, which describes only what crosses a stage boundary,
and the last stage's outgoing edge never transfers — so a head declares no shapes.

### Head configuration

Heads are selected by a closed, discriminated union resolved by a free `build_decoder_head(config, *, backbone) ->
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

A bespoke head a user writes for their own model is a `TaskHead` instance passed directly to `DecoderWithHead` or
`DecoderWithHeads`, bypassing the union entirely.

### The multi-head model

One generic class composes a backbone with a mapping of *prebuilt* heads, generic over the backbone type so the backbone
keeps its precise type for `parallelize_*`, and over the head IO so the composition's own IO stays typed. It does not
build heads — that is the factory's job — so the class stays free of the config union:

```python
class DecoderWithHeads(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor],
        SequenceHeadsShared[THeadShared], SequenceHeadsOutput[THeadOutput],
    ],
    Generic[TBackbone, THeadShared, THeadOutput],
):
    def __init__(
        self, backbone: TBackbone, heads: Mapping[str, TaskHead[THeadShared, THeadOutput]], stage: PipelineStageInfo
    ):
        self.model = backbone                                   # FQN: model.*
        self._stage = stage
        if stage.is_current_stage_last:
            self.heads = nn.ModuleDict(...)                     # FQN: heads.<name>.*

    def forward(self, inputs, shared: SequenceHeadsShared[THeadShared]):
        model_outputs = self.model(inputs, shared.sequence)
        if not self._stage.is_current_stage_last:
            return model_outputs
        return {name: head(model_outputs.hidden_states, shared.heads[name]) for name, head in self.heads.items()}
```

The `SharedInput` carries both halves of the composition — `shared.sequence` for the backbone, `shared.heads[name]` for
each head — so routing is by *name*, not by string keys in a flat bag. The mapping level is irreducible for an open head
set, but its *value* is not `Any`: it is the head IO the model was composed with, so a single-head model type-checks end
to end and a multi-head one names the union it actually produces:

```python
@dataclasses.dataclass
class SequenceHeadsShared(Generic[THeadShared]):
    sequence: SequenceShared
    heads: Mapping[str, THeadShared]

SequenceHeadsOutput: TypeAlias = Mapping[str, THeadOutput]
```

`reset_parameters` delegates to the backbone and the heads; `stage_transfer_spec` delegates to the backbone, since the
heads only run on the last stage, whose outgoing edge never transfers. The backbone (`self.model`) and the heads
(`self.heads`) are public, so a provider parallelizes each independently.

### The single-head model

Attaching one head is the common case, and with one head every part of the multi-head machinery is dead weight: there
is nothing to key, so a name has to be invented, threaded through the FQNs, and repeated by the task on both the input
and the output — `shared.heads["lm"]`, `results["lm"].logps` — to address the only head there is. That is a *different*
contract, not a specialization of the mapping one, so it gets its own class rather than a subclass of
`DecoderWithHeads`:

```python
class DecoderWithHead(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequenceHeadShared[THeadShared], THeadOutput,
    ],
    Generic[TBackbone, THead, THeadShared, THeadOutput],
):
    def __init__(self, backbone: TBackbone, head: THead, stage: PipelineStageInfo):
        self.model = backbone                                   # FQN: model.*
        self._stage = stage
        if stage.is_current_stage_last:
            self.head = head                                    # FQN: head.*

    def forward(self, inputs, shared: SequenceHeadShared[THeadShared]):
        model_outputs = self.model(inputs, shared.sequence)
        if not self._stage.is_current_stage_last:
            return model_outputs
        return self.head(model_outputs.hidden_states, shared.head)
```

The head is reached as `self.head`, its shared input arrives as `shared.head`, and the `PipelineOutput` *is* the head's
output — `ctx.pipeline_results.logps`, no key. `THead` is a parameter so `self.head` keeps its concrete type and the
per-head-type `parallelize_*_head` accepts it directly.

Three subclasses fix the head and build it, so composing the common model is one call:

```python
class DecoderForCausalLM(
    DecoderWithHead[TBackbone, SplitLanguageModellingHead, SequenceCausalLMHeadShared, SequenceCausalLMOutput]
):
    def __init__(self, backbone: TBackbone, stage: PipelineStageInfo):
        super().__init__(backbone, build_decoder_head(CausalLMHeadConfig(), backbone=backbone), stage)
```

`DecoderForClassification` and `DecoderForEmbedding` are the same shape, taking their head's config (`num_labels`,
`embedding_dim`, …) as a second positional argument. A model with two heads, or one that names its heads for its own
reasons, uses `DecoderWithHeads`.

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
out of the `TaskHead` module, in their own layer — and both stay **per head type**, because both are where heads are
expected to diverge (vocab parallelism for an LM head, say):

* **Sharding is per-head-type** — today every head is HSDP on the dense mesh, but a single universal function would
  have to be split the moment one head wants a different strategy, so the split is made up front. These are
  family-independent:

  ```python
  # d9d/module/parallelism/model/head.py
  def parallelize_causal_lm_head(head: SplitLanguageModellingHead, dist_context: DistributedContext) -> None: ...
  def parallelize_classification_head(head: ClassificationHead, dist_context: DistributedContext) -> None: ...
  def parallelize_embedding_head(head: EmbeddingHead, dist_context: DistributedContext) -> None: ...
  ```

* **Mapping is per-family *and* per-head-type** — a HuggingFace checkpoint carries exactly one head, whose parameter
  names are that HF architecture's own, so the mapper cannot be assembled from a family half and a head half. The
  existing hard-coded converter per (family, head) stays; what it gains is a `head_prefix`, telling it where in the
  composed model the HF head lands:

  ```python
  # d9d/module/model/qwen3_dense/huggingface.py
  def mapper_from_huggingface_qwen3_dense_for_causal_lm(params, *, head_prefix=SINGLE_HEAD_PREFIX): ...
  def mapper_from_huggingface_qwen3_dense_for_classification(params, *, head_prefix=SINGLE_HEAD_PREFIX): ...
  def mapper_from_huggingface_qwen3_dense_for_embedding(params): ...   # bare backbone: no head weights to place
  ```

  The default (`head.`) targets a single-head decoder. That parameter is what makes the multi-head use case work: a
  user initializing a two-head model from a single-head HuggingFace checkpoint passes the receiving head's prefix
  (`f"heads.{name}."`), and the others keep their initialization.

Because both composition classes expose the backbone (`self.model`) and the head(s) (`self.head` / `self.heads`), a
provider drives each independently — the per-family backbone routine on the backbone, then each head's own routine:

```python
def parallelize_model_stage(self, ctx):                                          # DecoderForCausalLM
    parallelize_qwen3_moe_model(ctx.dist_context, ctx.model.model, ctx.stage)    # backbone
    if ctx.stage.is_current_stage_last:
        parallelize_causal_lm_head(ctx.model.head, ctx.dist_context)             # head

def parallelize_model_stage(self, ctx):                                          # DecoderWithHeads
    parallelize_qwen3_moe_model(ctx.dist_context, ctx.model.model, ctx.stage)
    if ctx.stage.is_current_stage_last:
        parallelize_causal_lm_head(ctx.model.heads["lm"], ctx.dist_context)
        parallelize_classification_head(ctx.model.heads["cls"], ctx.dist_context)
```

The model-wrapper grid drops to `N` backbone routines plus the head-generic compositions; sharding and mapping keep
one entry per head type, which is what lets each evolve on its own.

## Usage

A provider builds the backbone and composes it with a head. For the common single-head case that is one call:

```python
backbone = Qwen3MoEModel(cfg.model, stage, hidden_states_snapshot_mode=..., enable_checkpointing=...)
model = DecoderForCausalLM(backbone, stage)
```

The task reads that head's output directly — `ctx.pipeline_results.logps`. A second head means dropping to
`DecoderWithHeads` and naming both; the task then reads both keys:

```python
heads = {
    "lm": build_decoder_head(CausalLMHeadConfig(), backbone=backbone),
    "cls": build_decoder_head(ClassificationHeadConfig(num_labels=3), backbone=backbone),
}
model = DecoderWithHeads(backbone, heads, stage)
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
  `parallelize_*_for_*` functions; providers compose a `DecoderFor*` subclass (or `DecoderWithHead`) for one head and
  `DecoderWithHeads` for several, and head sharding moves to the per-head-type `parallelize_*_head`. The
  `mapper_*_for_*` HuggingFace converters keep their names and gain a `head_prefix`. The backbone exposes
  `hidden_size` and its split-vocab layout as read-only properties; backbone params, its `forward`, and the
  per-family backbone `parallelize`/`mapper` routines are otherwise unchanged.
- **Parameter FQNs.** The backbone stays `model.*`; a single head moves `lm_head.* → head.*` (etc.) and heads of a
  multi-head model live under `heads.<name>.*`. Existing checkpoints need a one-pass key remap; the HuggingFace
  renames update accordingly.
- **IO types.** `SequenceCausalLMShared` / `SequencePoolingShared` (a backbone shared input bundled with one head's
  payload) are replaced by `SequenceHeadShared` / `SequenceHeadsShared` plus the per-head
  `SequenceCausalLMHeadShared` / `SequencePoolingHeadShared`. A single-head model's `PipelineOutput` is still that
  head's output dataclass, so `results.logps` keeps working; a multi-head model's is a mapping keyed by head name
  (`results["lm"].logps`). The per-head output dataclasses themselves are unchanged, but move to
  `d9d.module.block.head` alongside the heads that produce them.
- **Tests.** The suites under `test/d9d_test/modules/model/sequence/` (HF parity + state-dict round-trip across both
  families × all heads) update to the new construction and are the regression gate.

## Alternatives Considered

**Per-family thin wrappers (the earlier draft of this DEP).** Keep one wrapper per task per family, factoring shared glue
into three generic base classes each subclassed in ~6 lines. *Rejected:* cheaper copies, but the `3 × N` grid remains
(in all three axes) and a multi-head model still cannot be expressed. Its byte-stable-FQN requirement is what forced the
per-family subclass; lifting it (pre-1.0) is what unlocks composition.

**Single-head models as subclasses of `DecoderWithHeads`.** Express `DecoderForCausalLM` as a `DecoderWithHeads`
holding a one-entry mapping, defaulting the key to `"lm"`. *Rejected:* the composition would still be a mapping
everywhere it is observed — a `heads.lm.*` FQN, `shared.heads["lm"]` on the way in, `results["lm"]` on the way out —
so every caller pays for a name that exists only because the container needs one, and the substitution is unsound
anyway (the base promises a mapping output; a single-head model that returned one would defeat the purpose).
`DecoderWithHead` is a sibling, not a specialization: one head, no key, and the head's own output as the
`PipelineOutput`.

**A generic class that builds its own heads (`from_configs`).** Give `DecoderWithHeads` a classmethod that constructs an
arbitrary head mapping from a config mapping. *Rejected:* it couples the *generic* composition class to the head-config
union while buying nothing a caller cannot write in one line; `DecoderWithHeads` stays a pure composition of prebuilt
modules, so custom heads come in on equal footing. `DecoderWithHead` keeps that property; only its three subclasses
take the opposite trade, deliberately — each is bound to exactly one config type already, and building the head is
what makes composing the common model a single call.

**Universal head sharding (`parallelize_task_head`).** One function covering every head, since all of them are HSDP on
the dense mesh today. *Rejected:* the uniformity is a coincidence of the current head set, not a property of heads —
vocab parallelism for the LM head is the obvious next divergence — and callers would have to be rewritten when it
breaks. Three two-line functions cost nothing and pin the extension point where it belongs.

**Composable HF mappers (a family half plus a per-head-type half).** Assemble each converter from
`mapper_*_qwen3_dense` and a standalone `hf_mapper_for_lm_head(head, prefix)`. *Rejected:* it reads as though HF head
parameters were family-independent, which they are not — each HF architecture names its own head — and it makes every
call site re-derive a composition that has exactly one correct form. The hard-coded converter per (family, head) stays;
`head_prefix` is the only degree of freedom callers actually need.

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
