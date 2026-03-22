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

## Pipeline State

You may note that `batch` is only accessible in `build_forward_inputs(...)` method, but not in others. Don't worry!

There is an object for transferring any state between the **Task Lifecycle** stages, - it is called `PipelineState`.

```python
ctx.state["target"] = torch.tensor([1, 0, 1, 0], device="cuda")

# ...

metrics["accuracy"].update(ctx.state["target"])
```

The pipeline state will automatically shard and unshard data if needed.

You may read an [additional documentation](../internals/pipeline_state.md) for its internal behaviour.

## Example Implementation

```python
import torch

from d9d.core.dist_context import DistributedContext
from d9d.core.types import ScalarTree
from d9d.module.block.head import LM_IGNORE_INDEX
from d9d.loop.control import *

class SFTTask(TrainTask[dict[str, torch.Tensor]]):
    def __init__(self, dist_ctx: DistributedContext):
        self._dist_ctx = dist_ctx

    def build_forward_inputs(self, ctx: BuildForwardInputsContext) -> BuildForwardInputsResult:
        # ctx.batch contains the output of the Collator.

        # Save labels in state for access during loss computation later
        ctx.state["labels"] = ctx.batch["labels"]

        # Return inputs for model.forward()
        # inputs are only for the first pipeline stage
        # kwargs are the same for all the pipeline stages
        return BuildForwardInputsResult(
            inputs={
                "input_ids": ctx.batch["input_ids"]
            },
            kwargs={
                "labels": ctx.batch["labels"],
                "position_ids": ctx.batch["position_ids"]
            }
        )

    def dump_hparams(self) -> ScalarTree:
        return super().dump_hparams()

    def compute_loss(self, ctx: ComputeLossContext) -> ComputeLossResult:
        # Retrieve log_probs calculated by the model pipeline
        logps = ctx.pipeline_results["logps"]

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
