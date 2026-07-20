# User Tasks

A **Task** defines custom logic for a single train or inference step.

Each **Task** may implement `Stateful` protocol, so you may store some mutable state here.

## TrainTask

It is responsible for logging metrics, mapping batch inputs before they are fed into the model, and for computing the task loss function value.

**Init**: `create_metrics(...)`, `dump_hparams(...)`.

**Lifecycle**: 

1. `build_forward_inputs(...)` (will be called once) -> 
2. `compute_loss(...)` (will be called multiple times if pipelining is enabled - once for each pipeline microbatch) -> 
3. `update_metrics(...)` (will be called once).

**Exit**: `finalize(...)`.

**State Management**: `state_dict(...)`, `load_state_dict(...)`.

**Events Registration**: `register_events(...)` allows you to link specific custom methods to framework-wide [Event Hooks](./events.md).

## InferenceTask

The `InferenceTask` defines the logic for a single inference step. 

It is designed to handle the **forward-only** flow, processing the raw tensors synthesized by the model (e.g., logits, hidden states).

**Lifecycle**: 

1. `build_forward_inputs(...)` (called once) -> 
2. `process_outputs(...)` (called once per pipeline microbatch).

**Exit**: `finalize(...)`.

**State Management**: `state_dict(...)`, `load_state_dict(...)`.

**Events Registration**: `register_events(...)` allows hooking into the [Event Bus](./events.md) alongside regular execution.

## Task State

You may note that `batch` is only accessible in `build_forward_inputs(...)`, but not in the later stages. Don't worry!

The **state** carries side-data from `build_forward_inputs` to the later stages of the **same microbatch** — labels, masks, token counts, or anything else the model forward does not return but the loss or metrics need.

It is declared by `TState`, the last type parameter of `TrainTask` / `InferenceTask`. `TState` is any PyTree, so you can use whatever shape fits:

* a **dataclass** — when you want attribute access and strict typing;
* a **`TypedDict`** — when you prefer dict access but still want keys checked;
* a **plain `dict`** — for quick, untyped side-data;
* **`None`** — for tasks that carry nothing.

`build_forward_inputs` returns the state; `compute_loss` / `process_outputs` / `update_metrics` read it back, fully typed.

```python
class MyState(TypedDict):
    target: torch.Tensor

# in build_forward_inputs:
return BuildForwardInputsResult(input=..., shared=..., state=MyState(target=ctx.batch["target"]))

# later, in update_metrics / compute_loss:
metrics["accuracy"].update(ctx.state["target"])  # ctx.state is typed as MyState
```

Tensors stored in the state are detached from the autograd graph automatically, so caching them across the pipeline never keeps the graph alive.

## Example Implementation

A task's IO is typed by the same four PyTree roles the model pipeline uses (see [Pipeline
Parallelism](../models/pipeline_parallelism.md)). `TrainTask` is generic over
`[TBatch, TPipelineInput, TSharedInput, TPipelineOutput, TState]`:

* `TPipelineInput` — the `PipelineInput` fed to the **first** stage (built here).
* `TSharedInput` — the `SharedInput` broadcast to **every** stage.
* `TPipelineOutput` — the `PipelineOutput` produced by the **last** stage; read by attribute in `compute_loss`.

`build_forward_inputs` returns a `BuildForwardInputsResult` with `input` / `shared` / `state`
fields — no dict keys.

```python
import torch
from typing import TypedDict

from d9d.core.dist_context import DistributedContext
from d9d.core.types import ScalarTree
from d9d.module.block.head import LM_IGNORE_INDEX
from d9d.module.model.io import SequenceCausalLMOutput, SequenceCausalLMShared, SequenceInput, SequenceShared
from d9d.loop.control import *


class SFTState(TypedDict):  # it also could be a dataclass
    labels: torch.Tensor


class SFTTask(
    TrainTask[dict[str, torch.Tensor], SequenceInput, SequenceCausalLMShared, SequenceCausalLMOutput, SFTState]
):
    def __init__(self, dist_ctx: DistributedContext):
        self._dist_ctx = dist_ctx

    def build_forward_inputs(
        self, ctx: BuildForwardInputsContext
    ) -> BuildForwardInputsResult[SequenceInput, SequenceCausalLMShared, SFTState]:
        # ctx.batch contains the output of the Collator.

        # Return the PipelineInput, the SharedInput and the typed
        # side-data carried to loss computation for this same microbatch.
        return BuildForwardInputsResult(
            input=SequenceInput(input_ids=ctx.batch["input_ids"]),
            shared=SequenceCausalLMShared(
                sequence=SequenceShared(position_ids=ctx.batch["position_ids"]),
                labels=ctx.batch["labels"],
            ),
            state=SFTState(labels=ctx.batch["labels"]),
        )

    def dump_hparams(self) -> ScalarTree:
        return super().dump_hparams()

    def compute_loss(self, ctx: ComputeLossContext[SequenceCausalLMOutput, SFTState]) -> ComputeLossResult:
        # Retrieve log_probs calculated by the model pipeline
        logps = ctx.pipeline_results.logps

        # Calculate number of valid tokens (ignoring the -100 padding)
        # This is crucial for variable length batches.
        num_loss_tokens = (ctx.state["labels"] != LM_IGNORE_INDEX).sum()

        # Calculate average loss per valid token
        total_loss = logps.sum() / num_loss_tokens

        return ComputeLossResult(
            loss=total_loss,
            # loss_weight is used for gradient accumulation across the distributed world.
            # If batches have different token counts, we weigh the gradient
            # by token count to get a mathematical true average over the accumulation steps.
            loss_weight=num_loss_tokens / 1000
        )
```

::: d9d.loop.control.task
