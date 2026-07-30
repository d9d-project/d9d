---
DEP: 0011
Title: Multimodal Models
Author: Kirill Ryzhichkin @l1ghtsource
Status: Draft
Type: Feature
Created: 2026-07-28
---

# DEP-0011: Multimodal Models

## Abstract

d9d's model catalogue is text-only: every model consumes `SequenceInput` (token ids) and nothing
else. Modern frontier models (Qwen3-VL, Qwen3.5, Qwen3-Omni) are multimodal — a language backbone
consumes a token sequence in which some positions are placeholders filled with embeddings produced
by one or more *modality encoders* (a vision tower for images and videos, an audio tower, etc.).

This DEP introduces multimodality as a first-class entity in d9d:

1. **Packed media IO types** — `MediaSegments` and the `ModalityEncoder` trait, plus the
   catalogue-level `MultimodalSequenceInput` pipeline input.
2. **The merge convention** — `merge_media_embeddings` replaces placeholder token embeddings on
   the first pipeline stage; nothing downstream of the first stage changes.
3. **Vision building blocks** — patch embedding, packed bidirectional attention, 2D rotary
   embeddings, interpolated position embeddings, and a spatial patch merger under
   `d9d.module.block.vision`.
4. **Multimodal rotary embeddings (MRoPE)** — `MultimodalRotaryEmbeddingProvider` for 3D
   (temporal/height/width) position ids, plus dataset-side helpers to compute them.
5. **Varlen SDPA backends** — a variable-length counterpart to the DEP-0008 attention backend
   protocol, required for packed (non-rectangular) media attention.
6. **Data conventions** — collator contracts for packing media, placeholder masks, and the
   empty-media convention needed for collective-safety under FSDP.

The design is purely additive. The first consumer will be the Qwen3.5 model family (dense and
MoE), implemented in a follow-up (non-DEP) PR on top of these APIs.

## Motivation

Adding a multimodal model to d9d today would require ad-hoc answers to at least five architectural
questions, each of which becomes a de-facto convention the moment the first model ships:

* **Where do media tensors enter the pipeline?** Media inputs are variable-length and not
  batch-aligned (`features` is `(total_patches, dim)` across the whole microbatch), unlike
  everything the loop currently moves.
* **How do modality embeddings meet the token sequence** under pipeline parallelism, where the
  embedding table and the encoder only exist on the first stage?
* **How is attention computed over packed, non-rectangular media** (thousands of patches from
  images of different sizes) without padding?
* **Where are multimodal position ids computed?** HuggingFace computes 3D MRoPE indices inside the
  model on every forward; d9d's convention is that index arithmetic is dataset-side CPU work
  (precedent: causal-LM label shifting).
* **How do microbatches without media stay collective-safe?** FSDP all-gathers are lazy and
  triggered by module forward; if the vision tower runs on one rank of a shard group but not
  another, training deadlocks.

Answering these questions per-model would fragment the framework. Answering them once, here, gives
every future multimodal model (vision, audio, or both) a paved road, while preserving d9d's core
principle: models are explicit compositions of blocks, not configurations of an Uber-Module.

## Design Proposal

### Overview

A multimodal model in d9d is the explicit composition of three parts:

```mermaid
flowchart LR
    subgraph Stage 0
        E[Token Embeddings]
        V[Modality Encoder]
        M[merge_media_embeddings]
        L0[Decoder Layers 0..k]
        E --> M
        V --> M
        M --> L0
    end
    subgraph Stage 1..N
        L1[Decoder Layers k..n]
        H[Head]
        L1 --> H
    end
    L0 -- "SequenceTransfer (batch, seq, hidden)" --> L1
```

Everything modality-specific happens strictly on the first pipeline stage. After the merge, the
payload crossing stage boundaries is the existing `SequenceTransfer` — the pipelining engine
(DEP-0010 typed PyTree IO), schedules, and parallelization plans are untouched.

### 1. Media IO types

The media stream and the encoder trait live in `d9d.module.base` (they are model-independent
structural definitions, like `ModuleLateInit`):

```python
@dataclasses.dataclass
class MediaSegments:
    features: torch.Tensor   # (total_patches, feature_dim)
    grid_thw: torch.Tensor   # (num_segments, 3) - (temporal, height, width)


@typing.runtime_checkable
class ModalityEncoder(typing.Protocol):
    def __call__(self, media: MediaSegments) -> torch.Tensor:
        """-> (total_media_tokens, hidden_size), ordered segment-by-segment."""
```

The catalogue-level pipeline input joins the `Sequence*` family in `d9d.module.model.io`:

```python
@dataclasses.dataclass
class MultimodalSequenceInput:
    input_ids: torch.Tensor          # (batch, seq)
    media: MediaSegments
    media_token_mask: torch.Tensor   # (batch, seq), bool
```

Design points:

* **Packed, not per-sample.** `features` concatenates all media of the microbatch along dim 0.
  This matches varlen attention kernels, avoids padding entirely, and — because the loop never
  splits tensors (microbatches are formed by the collator) — requires no framework changes.
* **One stream per encoder.** Images and videos that share an encoder (the Qwen3-VL/Qwen3.5
  design) share one `MediaSegments` stream: an image is a segment with ``t == 1``, a video a
  segment with ``t > 1``. A model with several encoders (e.g. vision + audio) declares its own
  `PipelineInput` dataclass with one `MediaSegments` field per encoder — explicit composition, no
  generic "list of modalities" machinery.
* **Ordering contract.** Segments in `grid_thw` appear in the same order as their placeholder runs
  occur in the row-major flattened `input_ids`. The collator guarantees this; the model validates
  only the total count.
* **Shared inputs are reused.** `SequenceShared`/`SequenceCausalLMShared` stay as-is; a multimodal
  model simply carries `position_ids` of shape `(3, batch, seq)` in the same field (see §4).
* **`SequenceTransfer` is reused.** `stage_transfer_spec` derives shapes from `input_ids` exactly
  as today; media never crosses a stage boundary.

### 2. The merge convention

The merge is a small, reusable function in `d9d.module.block.embedding` (not a module — it has no
parameters):

```python
def merge_media_embeddings(
    token_embeddings: torch.Tensor,     # (batch, seq, hidden)
    media_token_mask: torch.Tensor,     # (batch, seq)
    media_embeddings: torch.Tensor,     # (total_media_tokens, hidden)
) -> torch.Tensor: ...
```

It validates `media_token_mask.sum() == media_embeddings.shape[0]` (fail-fast) and performs
`masked_scatter`. Gradients flow into both the token embedding table and the encoder. When the
mask selects no positions (the empty-media convention, §6), the media embeddings are instead
attached with a zero-valued contribution — this keeps the encoder inside the autograd graph so
every rank produces (zero) gradients for its parameters and gradient-sync collectives stay
aligned.

The first-stage forward of a multimodal model is then explicit and linear:

```python
if self._stage.is_current_stage_first:
    inputs = cast(MultimodalSequenceInput, inputs)
    hidden = self.embed_tokens(inputs.input_ids)
    media_embeds = self.media_encoder(inputs.media)
    hidden = merge_media_embeddings(hidden, inputs.media_token_mask, media_embeds)
```

**Pipeline load balancing.** The encoder's cost is accounted for with the existing
`pipeline_num_virtual_layers_pre` mechanism; no scheduler changes.

### 3. Vision building blocks (`d9d.module.block.vision`)

New blocks, all `nn.Module, ModuleLateInit`, composed freely by model packages (the assembled
vision tower lives in the model package, mirroring how decoder layers are assembled today):

| Block | Purpose |
|---|---|
| `PatchEmbedding` | `Conv3d`-based projection of raw `(total_patches, C·t·p·p)` features into hidden size; kernel = stride = `(temporal_patch_size, patch_size, patch_size)`. |
| `InterpolatedPositionEmbedding` | Learned absolute position table with bilinear interpolation to each segment's `(h, w)` grid, packed in spatial-merge block order. |
| `VisionRotaryEmbedding2D` | Produces per-token `(cos, sin)` for packed segments from `grid_thw` (row/column frequencies, HALF style — compatible with the existing `RotaryEmbeddingApplicator`). |
| `PackedVisionAttention` | Bidirectional multi-head attention over packed segments; consumes `cu_seqlens` so attention never crosses segment boundaries (see §5). |
| `GeluMLP` | Two-layer MLP with tanh-approximated GELU and biases (new sibling of `SwiGLU` in `block/ffn`). |
| `VisionBlock` | Pre-norm residual block: `LayerNorm → PackedVisionAttention → LayerNorm → GeluMLP`. |
| `SpatialPatchMerger` | Merges `spatial_merge_size²` neighboring patch embeddings and projects to the language hidden size (`LayerNorm → Linear → GELU → Linear`). |

A small helper `segment_cu_seqlens(grid_thw)` derives the attention segment boundaries (one
attention segment per temporal frame) once per forward — cheap integer arithmetic on device.

Configuration follows repo rules: pydantic parameter models at the model boundary, plain
constructor arguments for blocks.

Deliberately **not** included: a generic `VisionEncoder` module. Each model package assembles its
tower from these blocks explicitly, exactly as decoder layers are assembled from attention/FFN
blocks today. Qwen-family towers are near-identical; the model packages clone the assembly, which
is the established catalogue pattern (`qwen3_dense` vs `qwen3_moe`).

### 4. Multimodal rotary embeddings (MRoPE)

Two additions:

**Provider** (`d9d.module.block.positional`): `MultimodalRotaryEmbeddingProvider`, a sibling of
`RotaryEmbeddingProvider`:

```python
class MultimodalRotaryEmbeddingProvider(nn.Module, ModuleLateInit):
    def __init__(
        self,
        rope_base: int,
        rope_dim: int,
        max_position_ids: int,
        mrope_section: tuple[int, int, int],
        interleaved: bool = True,
        rope_scaling: RopeScaling | None = None,
    ) -> None: ...

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: (3, batch, seq) -> (cos, sin), each (batch, seq, rope_dim)."""
```

It caches the same cos/sin tables as the existing provider, gathers per-axis rows for the T/H/W
position planes, and combines them per `mrope_section` using a precomputed per-frequency plane
selector (interleaved `[THWTHW...TT]` layout used by Qwen3-VL/Qwen3.5, or the chunked Qwen2-VL
layout). The output plugs into the existing `RotaryEmbeddingApplicator` and
`GroupedQueryAttention.forward(position_embeddings=...)` unchanged.

For text-only content, `position_ids[0] == position_ids[1] == position_ids[2]` makes MRoPE
numerically identical to standard RoPE — this equivalence is pinned by a unit test.

**Dataset-side helpers** (`d9d.dataset`): computing the 3D indices is index arithmetic over
placeholder runs and `grid_thw`, so per repo convention (precedent: label shifting) it happens on
CPU in the collator:

```python
def compute_multimodal_position_ids(
    input_ids: torch.Tensor,            # (seq,)
    media_token_mask: torch.Tensor,     # (seq,)
    grid_thw: torch.Tensor,             # (num_segments, 3)
    spatial_merge_size: int,
) -> torch.Tensor:                      # (3, seq)
    ...
```

Models never recompute positions; the `SharedInput` already carries them to every stage.

### 5. Varlen SDPA backends

Packed media attention operates on `(total_tokens, heads, head_dim)` with `cu_seqlens`, which the
DEP-0008 `SdpaBackend` protocol cannot express. Rather than widening the existing protocol with
optional arguments, we add a parallel varlen trait in `d9d.module.block.attention.sdpa`:

```python
class VarlenSdpaBackend(Protocol):
    def __call__(
        self,
        query_states: torch.Tensor,     # (total_tokens, n_q_heads, head_dim)
        key_states: torch.Tensor,       # (total_tokens, n_kv_heads, head_dim)
        value_states: torch.Tensor,
        cu_seqlens: torch.Tensor,       # (num_segments + 1,), int32
        max_seqlen: int,
        is_causal: bool,
        scale: float,
    ) -> torch.Tensor: ...
```

with `build_varlen_sdpa_backend(params, backend_config)` following DEP-0008 exactly: the same
discriminated-union config classes, an env-var override (`D9D_BACKEND_AUTO_SDPA_VARLEN`), and
capability-based auto-detection:

* `flash_attention_4` → existing `d9d.kernel.flash_attn.flash_attn_varlen_func`;
* `flash_attention_2` → `flash_attn.flash_attn_varlen_func`;
* `torch` → torch SDPA over a block-diagonal mask built from `cu_seqlens` (correct, quadratic in
  the total packed length — the dependency-free fallback).

The `eager` config is rejected with a clear error (the torch backend is the reference fallback).
`PackedVisionAttention` takes an optional `AnySdpaBackendConfig` exactly as `GroupedQueryAttention`
does today.

### 6. Data conventions

**Collator contract** (documented in `docs/models/multimodality.md`):

1. Media features of all samples of a microbatch are concatenated into one `MediaSegments`, in
   row-major placeholder order.
2. `media_token_mask` marks placeholder positions; its `True`-count equals the encoder's output
   token count (`grid_thw` product sum divided by `spatial_merge_size²` for vision).
3. `position_ids` `(3, batch, seq)` are computed on CPU via `compute_multimodal_position_ids`.

**Empty-media convention.** Two constraints force media fields to always be present:

* DEP-0010 requires every microbatch in a pack to share the same `PipelineInput` PyTree structure
  — `media: None` in one microbatch and `media: MediaSegments` in another is rejected by the
  executor.
* FSDP all-gathers are lazily triggered by module forward. If the tower is skipped on a
  no-media microbatch on one rank of a shard group while another rank runs it, the all-gather
  collective deadlocks.

Therefore: *the modality encoder runs on every microbatch*. Text-only microbatches carry one dummy
segment (a single minimal merge block, `grid_thw = [[1, s, s]]` with `s = spatial_merge_size`)
whose output is attached by `merge_media_embeddings` with a zero-valued contribution (the
`media_token_mask` is all-`False`). The collator helper `pad_empty_media(media, ...) ->
MediaSegments` implements this. The dummy contributes zero gradient signal but keeps every
collective and the autograd graph structurally identical across ranks and microbatches.

### 7. Parallelism

Nothing new is required:

* The vision tower and merge live on the first stage only; the tower is parallelized with the
  existing `parallelize_hsdp` in the model's companion `parallelize_*` function.
* Expert parallelism (MoE backbones) is unaffected.
* TP and CP remain unsupported by catalogue models (`raise ValueError`), matching the current
  qwen3 plans. CP over a merged multimodal sequence is explicitly out of scope; it will arrive
  with the general CP design and its own DEP.

### 8. Testing

Per DEP-0004 (task-centric harness):

* **Block tier** (`local`): parity tests of the assembled vision tower against the HuggingFace
  `Qwen3_5MoeVisionModel` (forward + backward + mapped gradients, with a test-local state
  mapper); MRoPE parity against `Qwen3VLTextRotaryEmbedding`; the MRoPE-equals-RoPE-on-text
  equivalence; varlen backends against a per-segment SDPA reference; unit tests for
  `merge_media_embeddings`.
* **Data tier** (`local`): unit tests for `compute_multimodal_position_ids` (text-only, image,
  video, count-mismatch rejection) and `pad_empty_media`.
* **Model tier** (in the follow-up model PR): a new task directory
  `test/d9d_test/modules/model/multimodal/causal_lm/` with its own `batch.py` (synthetic packed
  images + placeholders), `catalogue.py`, `test_hf.py` (`local`) and `test_distributed.py`
  (`distributed`, reusing `MESHES_FOR_MODEL_TESTS`). Adding a future multimodal architecture then
  requires only a registry entry.

### 9. Documentation

* `docs/models/multimodality.md` — the architecture, IO types, merge and data conventions.
* `docs/models/modules/vision.md` — the vision block reference (mkdocstrings).
* Extended sections in the `attention` (varlen backends), `positional` (MRoPE), `embedding`
  (merge), `ffn` (GeluMLP) and `dataset` pages; `docs/toc.md` and `zensical.toml` nav updates.

## Usage

A model package composes the pieces explicitly (abridged):

```python
class MyMultimodalForCausalLM(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[
        MultimodalSequenceInput, SequenceTransfer[torch.Tensor], SequenceCausalLMShared, SequenceCausalLMOutput
    ],
):
    def __init__(self, params, stage, ...):
        ...
        if stage.is_current_stage_first:
            self.embed_tokens = SplitTokenEmbeddings(...)
            self.visual = MyVisionTower(params.vision)   # assembled from d9d.module.block.vision
        self.rope_provider = MultimodalRotaryEmbeddingProvider(..., mrope_section=params.mrope_section)

    def forward(self, inputs, shared):
        if self._stage.is_current_stage_first:
            inputs = cast(MultimodalSequenceInput, inputs)
            hidden = self.embed_tokens(inputs.input_ids)
            hidden = merge_media_embeddings(hidden, inputs.media_token_mask, self.visual(inputs.media))
        else:
            hidden = inputs.hidden_states
        rope = self.rope_provider(shared.sequence.position_ids)   # (3, batch, seq)
        ...
```

The task builds inputs from the collator output:

```python
return BuildForwardInputsResult(
    input=MultimodalSequenceInput(
        input_ids=batch["input_ids"],
        media=MediaSegments(features=batch["media_features"], grid_thw=batch["media_grid_thw"]),
        media_token_mask=batch["media_token_mask"],
    ),
    shared=SequenceCausalLMShared(
        sequence=SequenceShared(position_ids=batch["position_ids"]),  # (3, batch, seq)
        labels=batch["labels"],
    ),
    state=...,
)
```

## Backward Compatibility

Fully additive: new IO dataclasses, a new base trait, new blocks, a new positional provider, a new
varlen backend factory, and dataset helpers. No existing public API changes; existing text-only
models and training scripts are unaffected.

## Alternatives Considered

1. **A generic `MultimodalModel` base class / Uber-Module** that owns encoders, merging, and the
   backbone behind configuration flags. Rejected: violates the explicit-composition principle; the
   catalogue precedent (dense/MoE packages duplicating `model.py`) shows composition-by-cloning is
   the intended trade-off.
2. **Vision tower as a dedicated pipeline stage type.** Rejected: it complicates schedules and
   buffer specs (media would cross P2P), while `pipeline_num_virtual_layers_pre` already balances
   first-stage cost with zero engine changes.
3. **Per-sample media (lists of image tensors / nested PyTrees).** Rejected: padding-free packed
   layout matches varlen kernels; per-sample structure would vary across microbatches, violating
   the DEP-0010 uniform-structure rule, and would force padding or ragged handling everywhere.
4. **Computing MRoPE position ids inside the model** (HuggingFace behavior). Rejected: it is pure
   index arithmetic, recomputed identically on every stage and every forward; d9d's convention is
   dataset-side CPU preprocessing (precedent: label shifting), keeping stages free of redundant
   work.
5. **Widening `SdpaBackend` with optional `cu_seqlens` arguments** instead of a separate varlen
   protocol. Rejected: it would make every existing backend partially-implemented (masks vs
   varlen are mutually exclusive in most kernels) and turn a structural trait into a kitchen-sink
   signature. Two small protocols keep each contract total.
6. **Allowing `media: None` for text-only microbatches.** Rejected: breaks the DEP-0010 structural
   uniformity rule within a pack and creates FSDP collective hazards; the dummy-segment convention
   is cheap and keeps the graph rank-uniform.
