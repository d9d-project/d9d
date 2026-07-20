---
DEP: 0010
Title: Typed PyTree IO for Pipelining and Tasks
Author: Maksim Afanasyev @mrapplexz
Status: Draft
Type: Feature
Created: 2026-07-10
---

# Typed PyTree IO for Pipelining and Tasks

## Abstract

The pipelining engine and the `TrainTask`/`InferenceTask` APIs move all data across pipeline
boundaries as `dict[str, torch.Tensor]`. A single dict type is overloaded to mean four distinct
things — the input to the first stage, the payload transferred between adjacent stages, the
output of the last stage, and the arguments broadcast to every stage — with no type-level
distinction. String keys simultaneously act as `forward` parameter names (`**inputs` splat),
the P2P wire ordering (sorted-by-name pairing), the gradient-routing keys, and the loss field
names.

This DEP replaces stringly-typed dict IO with four explicitly-named, generic PyTree types.
Models declare their IO as dataclasses; transport becomes structure-driven instead of
name-driven; and the four roles become distinct, type-checked entities.

## Motivation

The overloading described above produces three concrete failures, anchored in today's code:

- **Roles documented in prose.** `BuildForwardInputsResult` must explain input vs
  kwargs vs state in its docstring because the signatures cannot.
- **Boilerplate `forward`.** One class serves every stage position, so `forward` lists every
  tensor as `... | None = None` and branches on which args are `None` — an implicit invariant.
- **Confusing spec inference.** `ModuleSupportsPipelining` requires two methods returning
  `dict[str, TensorSpec]` that duplicate most of their logic, and the untyped dict cannot be
  checked against the actual transfer structure.

## Design Proposal

### The four roles, named

| Role                                                        | Type             | Transferred over P2P? |
|-------------------------------------------------------------|------------------|-----------------------|
| Input to the first stage (built by the task)                | `PipelineInput`  | no                    |
| Payload between adjacent stages (out of *N* == in of *N+1*) | `StageTransfer`  | yes                   |
| Output of the last stage (to loss/result callback)          | `PipelineOutput` | no                    |
| Value passed to *every* stage, rebuilt locally per rank     | `SharedInput`    | no                    |

Each is an arbitrary PyTree; dataclasses are the recommended form. Only `StageTransfer` needs a
`TensorSpec` — it is the only thing crossing the wire, so only its receive buffer must be sized
ahead of the forward pass.

### Model-facing contract

A stage knows its position (`PipelineStageInfo` at construction), so it branches on explicit
position, never on `None`-sniffing. `ModuleSupportsPipelining` becomes generic over the four
types:

```python
class StageBoundary(enum.Enum):
    incoming = "incoming"   # StageTransfer received from stage N-1
    outgoing = "outgoing"   # StageTransfer sent to stage N+1


@typing.runtime_checkable
class ModuleSupportsPipelining(
    typing.Protocol[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]
):
    def forward(
        self,
        inputs: TPipelineInput | TStageTransfer,   # PipelineInput iff first stage, else StageTransfer
        shared: TSharedInput,
    ) -> TStageTransfer | TPipelineOutput:         # PipelineOutput iff last stage, else StageTransfer
        ...

    def stage_transfer_spec(
        self, pipeline_input: TPipelineInput, boundary: StageBoundary
    ) -> PyTree[TensorSpec]:
        ...
```

The spec must be structurally identical to `TStageTransfer`.

### Structure-driven transport

Because stage *N*'s `outgoing` and stage *N+1*'s `incoming` are the **same dataclass type**,
`pytree.tree_flatten` yields identical leaf orderings on both ends. Sender and receiver agree on
wire order by construction, so no shape/name handshake is needed.

- **Send**: flatten the outgoing `StageTransfer`; `isend` leaves in order.
- **Receive**: flatten `stage_transfer_spec(..., incoming)`
  to ordered `TensorSpec` leaves + treespec; allocate a buffer per leaf; `tree_unflatten` back
  into a `StageTransfer` for `forward`.
- **Backward** already uses `d9d.core.pytree`.
- **Forward**: drop the `**inputs` splat and the `Mapping` check; call `module(inputs, shared)`.

### Downstream API changes

- `BuildForwardInputsResult` → `Generic[TPipelineInput, TSharedInput, TState]` with
  `input`/`shared`/`state` (was `inputs`/`kwargs` dicts).
- `ComputeLossContext.pipeline_results` / `ProcessOutputsContext.pipeline_results` →
  `TPipelineOutput`.
- `PipelineLossFn` / `PipelineResultFn` take `TPipelineOutput`.
- `PipelineSchedule.step` generalizes to per-microbatch `TPipelineInput`/`TSharedInput`.
- The four type parameters thread through the protocol, `PipelineSchedule`, task base classes,
  and loss/result contexts. Pure tensor-shuffling internals stay on `PyTree`/leaf lists.

## Usage

```python
TLeaf = TypeVar("TLeaf")

@dataclasses.dataclass
class CausalLMInput:                       # PipelineInput
    input_ids: torch.Tensor

@dataclasses.dataclass
class Qwen3Transfer(Generic[TLeaf]):       # StageTransfer, generic over its leaf type
    hidden_states: TLeaf

@dataclasses.dataclass
class CausalLMShared:                      # SharedInput
    position_ids: torch.Tensor
    labels: torch.Tensor | None = None

@dataclasses.dataclass
class CausalLMOutput:                      # PipelineOutput
    logps: torch.Tensor


class Qwen3DenseForCausalLM(
    nn.Module, ModuleLateInit,
    ModuleSupportsPipelining[CausalLMInput, Qwen3Transfer[torch.Tensor], CausalLMShared, CausalLMOutput],
):
    def forward(
        self, inputs: CausalLMInput | Qwen3Transfer[torch.Tensor], shared: CausalLMShared
    ) -> CausalLMOutput | Qwen3Transfer[torch.Tensor]:
        if self._stage.is_current_stage_first:
            hidden = self.model.embed(inputs.input_ids)
        else:
            hidden = inputs.hidden_states
        hidden = self.model.layers(hidden, shared)
        if self._stage.is_current_stage_last:
            return CausalLMOutput(logps=self.lm_head(hidden, labels=shared.labels))
        else:
            return Qwen3Transfer(hidden_states=hidden)

    def stage_transfer_spec(
        self, pipeline_input: CausalLMInput, boundary: StageBoundary
    ) -> Qwen3Transfer[TensorSpec]:
        b, s = pipeline_input.input_ids.shape
        return Qwen3Transfer(hidden_states=TensorSpec(shape=(b, s, self._hidden_size), dtype=self._dtype))
```

The task then builds `PipelineInput`/`SharedInput` and reads `ctx.pipeline_results.logps` by
attribute — no dict keys.

## Backward Compatibility

Breaks public API.

## Alternatives Considered

1. **Keep dicts, add name-matching validation.** Leaves the
   overloaded type, the splat, and the six-optional `forward`. Symptom, not disease.
2. **`TypedDict` instead of dataclasses.** Still a runtime `dict`,
   cannot express the `PipelineInput | StageTransfer` union cleanly.
3. **Trace `forward` under `FakeTensorMode` to derive transfer shapes**, dropping
   `stage_transfer_spec` entirely. Rejected: fake-tensor tracing still executes the full Python
   body of every stage (layer loops, routing branches) on CPU, re-run on each shape change — far
   more expensive than the cheap shape arithmetic an explicit spec method does. The explicit
   method is the point: it sizes buffers without running the body.
