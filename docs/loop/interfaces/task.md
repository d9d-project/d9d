# User Tasks

## About

A **task** defines the custom logic of a single train or inference step. A task implements the `Stateful` protocol, so it can store mutable state that is saved in checkpoints.

## TrainTask

A `TrainTask` builds the model inputs from each microbatch, computes the loss and updates the metrics.

**Init**: `create_metrics(...)`, `dump_hparams(...)`.

**Lifecycle**, for each microbatch of a step:

1.  `build_forward_inputs(...)` ->
2.  `compute_loss(...)` ->
3.  `update_metrics(...)`.

**Exit**: `finalize(...)`.

**State Management**: `state_dict(...)`, `load_state_dict(...)`.

**Events Registration**: `register_events(...)` links your methods to [event hooks](./events.md).

## InferenceTask

An `InferenceTask` defines the logic of a single inference step. It handles the forward-only flow and processes the outputs of the model (e.g. logits, hidden states).

**Lifecycle**, for each microbatch of a step:

1.  `build_forward_inputs(...)` ->
2.  `process_outputs(...)`.

**Exit**: `finalize(...)`.

**State Management**: `state_dict(...)`, `load_state_dict(...)`.

**Events Registration**: `register_events(...)` links your methods to the [event bus](./events.md).

## Task State

The raw `batch` is available only in `build_forward_inputs(...)`. The **state** carries side-data from `build_forward_inputs` to the later calls for the same microbatch. Use it for labels, masks, token counts or anything else the loss or the metrics need but the model does not return.

`TState`, the last type parameter of `TrainTask` / `InferenceTask`, declares the state. `TState` can be any PyTree:

*   a dataclass, for attribute access and strict typing;
*   a `TypedDict`, for dict access with checked keys;
*   a plain `dict`, for quick, untyped side-data;
*   **`None`**, for tasks that carry nothing.

`build_forward_inputs` returns the state. `compute_loss`, `process_outputs` and `update_metrics` read it back, fully typed.

```python
class MyState(TypedDict):
    target: torch.Tensor

# In build_forward_inputs:
return BuildForwardInputsResult(input=..., shared=..., state=MyState(target=ctx.batch["target"]))

# Later, in update_metrics / compute_loss:
ctx.metrics["accuracy"].update(ctx.state["target"])  # ctx.state is typed as MyState
```

Tensors stored in the state are detached from the autograd graph, so the cached state never keeps the graph alive.

## Task Input and Output Types

The task I/O uses the same PyTree roles as the model pipeline (see [Pipeline Parallelism](../../models/pipeline_parallelism.md)). `TrainTask` is generic over `[TBatch, TPipelineInput, TSharedInput, TPipelineOutput, TState]`:

*   `TBatch`: the raw microbatch produced by the data stream.
*   `TPipelineInput`: the `PipelineInput` fed to the first stage.
*   `TSharedInput`: the `SharedInput` passed to every stage.
*   `TPipelineOutput`: the `PipelineOutput` produced by the last stage and read in `compute_loss`. For a single-head model, it is the output of that head. For a model with several named heads, it holds the output of each head, keyed by head name.

`build_forward_inputs` returns a `BuildForwardInputsResult` with the `input`, `shared` and `state` fields.

## Usage

```python
from typing import TypedDict

import torch

from d9d.core.dist_context import DistributedContext
from d9d.core.types import ScalarTree
from d9d.loop.control import (
    BuildForwardInputsContext,
    BuildForwardInputsResult,
    ComputeLossContext,
    ComputeLossResult,
    TrainTask,
)
from d9d.module.block.head import LM_IGNORE_INDEX, SequenceCausalLMHeadShared, SequenceCausalLMOutput
from d9d.module.model.io import SequenceHeadShared, SequenceInput, SequenceShared


class SFTState(TypedDict):  # It can also be a dataclass
    labels: torch.Tensor


class SFTTask(
    TrainTask[
        dict[str, torch.Tensor],
        SequenceInput,
        SequenceHeadShared[SequenceCausalLMHeadShared],
        SequenceCausalLMOutput,
        SFTState,
    ]
):
    def __init__(self, dist_ctx: DistributedContext):
        self._dist_ctx = dist_ctx

    def build_forward_inputs(
        self, ctx: BuildForwardInputsContext
    ) -> BuildForwardInputsResult[SequenceInput, SequenceHeadShared[SequenceCausalLMHeadShared], SFTState]:
        # ctx.batch contains the output of the collator.
        # The SharedInput routes position IDs to the backbone and labels to the head.
        return BuildForwardInputsResult(
            input=SequenceInput(input_ids=ctx.batch["input_ids"]),
            shared=SequenceHeadShared(
                sequence=SequenceShared(position_ids=ctx.batch["position_ids"]),
                head=SequenceCausalLMHeadShared(labels=ctx.batch["labels"]),
            ),
            state=SFTState(labels=ctx.batch["labels"]),
        )

    def dump_hparams(self) -> ScalarTree:
        return super().dump_hparams()

    def compute_loss(self, ctx: ComputeLossContext[SequenceCausalLMOutput, SFTState]) -> ComputeLossResult:
        logps = ctx.pipeline_results.logps

        # Count the valid tokens (labels that are not padding). The loss of a variable-length batch needs it.
        num_loss_tokens = (ctx.state["labels"] != LM_IGNORE_INDEX).sum()

        # Average loss per valid token.
        total_loss = logps.sum() / num_loss_tokens

        return ComputeLossResult(
            loss=total_loss,
            # loss_weight weights this microbatch in the gradient average across microbatches and ranks.
            # Weighting by token count gives the true per-token average when token counts differ.
            loss_weight=num_loss_tokens / 1000,
        )
```

## API Reference

::: d9d.loop.control.task
