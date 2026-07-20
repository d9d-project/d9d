# Pipeline Parallelism

## The d9d Approach

d9d implements a modern, highly modular pipelining engine designed for performance, stability and customization.

### Dynamic Shapes & Algorithmic Shape Inference

To run P2P (Point-to-Point) communication, the receiver must know the shape of the incoming tensor to pre-allocate buffers. d9d asks your model to implement a lightweight protocol (`ModuleSupportsPipelining`) to calculate the shape of the payload transferred between stages mathematically, without performing a heavy forward pass or doing a distributed graph tracing.

This allows supporting **Dynamic Shapes** (e.g., varying sequence lengths) efficiently across runs.

### Construction Consistency (No Patching)
A common anti-pattern in distributed training is "Instantiate-then-Delete": creating a huge model on CPU/Meta device and then hacking it apart `del model.layers[N:]`. 

We reject this pattern because of:

1.  **Fragility**: Changes to model architecture require changes to the external slicing script.
2.  **Leaky Abstractions**: Forward methods become full of `if self.layer is not None`.
3.  **Invalid States**: The model object exists in a "zombie" state until sliced.

In d9d, models are **Pipeline-Aware**. Each pipeline rank constructs **only** the sub-graph it owns. The object returned is compliant, complete, and valid immediately.

## Making Models Compatible

### The Four IO Roles

A pipelined model moves data across stage boundaries as four **explicitly-named, generic PyTree
types** (dataclasses are the recommended form):

| Role             | Meaning                                                     | Crosses P2P? |
|------------------|-------------------------------------------------------------|--------------|
| `PipelineInput`  | Input to the **first** stage (built by the task)            | no           |
| `StageTransfer`  | Payload between adjacent stages (out of *N* == in of *N+1*) | **yes**      |
| `PipelineOutput` | Output of the **last** stage (to loss / result callback)    | no           |
| `SharedInput`    | Value passed to **every** stage, rebuilt locally per rank   | no           |

Only `StageTransfer` crosses the wire, so it is the only role that needs a `TensorSpec`.

### The Protocol

To use Pipeline Parallelism, your model implements
`d9d.pipelining.api.ModuleSupportsPipelining[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]`:

* **`forward(inputs, shared)`** — `inputs` is the `PipelineInput` on the first stage and the incoming
  `StageTransfer` otherwise; it returns the outgoing `StageTransfer` on non-last stages and the
  `PipelineOutput` on the last stage. The stage knows its position from the `PipelineStageInfo` it
  received at construction, so it branches on `is_current_stage_first` / `is_current_stage_last`
  explicitly.
* **`stage_transfer_spec(pipeline_input, boundary)`** — returns a PyTree **structurally identical to
  `StageTransfer`** with every tensor leaf replaced by a `TensorSpec`. The `boundary`
  (`StageBoundary.incoming` / `outgoing`) selects which inter-stage edge to size.

Because stage *N*'s `outgoing` transfer and stage *N+1*'s `incoming` transfer are the **same
dataclass type**, `pytree.tree_flatten` yields identical leaf orderings on both ends. Sender and
receiver therefore agree on the wire order **by construction** — no name/shape handshake is needed.

### Example

Below is a skeleton of a Transformer-like model implemented for d9d pipelining.

```python
import dataclasses
import torch
from torch import nn
from d9d.pipelining.api import (
    ModuleSupportsPipelining,
    PipelineStageInfo,
    StageBoundary,
    TensorSpec,
    distribute_layers_for_pipeline_stage,
)


@dataclasses.dataclass
class MyInput:              # PipelineInput
    input_ids: torch.Tensor


@dataclasses.dataclass
class MyTransfer:           # StageTransfer (the only role crossing P2P)
    hidden_states: torch.Tensor


@dataclasses.dataclass
class MyShared:             # SharedInput (broadcast to every stage)
    position_ids: torch.Tensor


@dataclasses.dataclass
class MyOutput:             # PipelineOutput
    logits: torch.Tensor


class MyModelChunk(
    nn.Module,
    ModuleSupportsPipelining[MyInput, MyTransfer, MyShared, MyOutput],
):
    def __init__(self, stage: PipelineStageInfo, config):
        super().__init__()
        self.stage = stage
        self.config = config

        # 1. Determine what layers live here
        self.start_layer, self.end_layer = distribute_layers_for_pipeline_stage(
            config.n_layers, num_virtual_layers_pre=1, num_virtual_layers_post=1, stage=stage
        )

        # 2. Build sub-modules (using ModuleDict - for compatibility)
        self.layers = nn.ModuleDict({
            str(layer): TransformerBlock(...)
            for layer in range(self.start_layer, self.end_layer)
        })

        # Only build embeddings on first stage
        if stage.is_current_stage_first:
            self.embed = nn.Embedding(...)

        # Only build head on last stage
        if stage.is_current_stage_last:
            self.head = nn.Linear(...)

    def forward(self, inputs: MyInput | MyTransfer, shared: MyShared) -> MyTransfer | MyOutput:
        # Branch on the stage position, never on which argument is None.
        if self.stage.is_current_stage_first:
            x = self.embed(inputs.input_ids)
        else:
            x = inputs.hidden_states

        # Run local layers
        for layer_idx in range(self.start_layer, self.end_layer):
            x = self.layers[str(layer_idx)](x)

        # Last stage produces the PipelineOutput; everyone else produces a StageTransfer.
        if self.stage.is_current_stage_last:
            return MyOutput(logits=self.head(x))
        return MyTransfer(hidden_states=x)

    # --- Protocol Implementation ---
    # Receives a single representative microbatch of PipelineInput (not a global batch), so shapes are
    # used as-is, and returns a StageTransfer of TensorSpec descriptors — no tensors are allocated.
    # The engine never calls this for the first stage's `incoming` nor the last stage's `outgoing`
    # edge, so terminal boundaries need no special-casing.
    def stage_transfer_spec(self, pipeline_input: MyInput, boundary: StageBoundary) -> MyTransfer:
        micro_batch_size, seq_len = pipeline_input.input_ids.shape
        return MyTransfer(
            hidden_states=TensorSpec(
                shape=(micro_batch_size, seq_len, self.config.hidden_dim), dtype=torch.bfloat16
            )
        )
```

## Using the Pipeline

### Supported Schedules

| Example JSON                                                          | Description                                                                                                                                                             |
|:----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `{"schedule": "inference"}`                                           | Configuration for inference-only pipeline execution. Runs all forward passes sequentially without any backward passes.                                                  |
| `{"schedule": "gpipe"}`                                               | Standard GPipe execution. Assumes a single stage per rank and processes all microbatches for the forward pass before switching to the backward pass.                    |
| `{"schedule": "looped_bfs", "num_stages_per_rank": 2}`                | Looped Breadth-First Search execution. Supports multiple stages per rank (virtualization) and executes all work for a specific stage before moving to the next.         |
| `{"schedule": "1f1b", "num_stages_per_rank": 1, "zero_bubble": true}` | Interleaved 1F1B and Interleaved Zero Bubble execution. Supports multiple stages per rank. Handles sharding backward passes to dI and dW when `zero_bubble` is enabled. |
| `{"schedule": "zero_bubble_v"}`                                       | Zero Bubble V (ZBV) execution. A specialized V-shape topology schedule that splits backward passes into Input and Weight gradients. Requires exactly 2 stages per rank. |
| `{"schedule": "dual_pipe_v"}`                                         | DualPipeV execution. A bidirectional pipeline schedule for high-throughput training using V-shape topology and reciprocal forward/backward scheduling.                  |

### Microbatches and packs

Pipelining consumes a **pack**: a sequence of ready microbatches for one step. The pack length (and therefore the batch size) may vary from step to step. Buffers are sized
**per microbatch** — each stage infers shapes for every microbatch in the pack independently — so the
microbatches within a single pack may also differ in shape.
The schedule recompiles its program when the microbatch count changes and reallocates buffers when any
microbatch's shape changes.

### Usage within the Trainer

Pipelining is available in the [Trainer](../loop/train.md) framework. When configuring the Trainer, simply provide an `AnyPipelineScheduleConfig` in your training arguments. The Trainer handles the construction of the schedule and the distribution of layers automatically.

### Advanced - Manual Usage

If you want to use pipelining outside the Trainer (e.g., custom loops), you use the `build_schedule` factory.

The `build_schedule` function requires a **Model Provider** logic. Instead of passing an instantiated model, you pass a function that accepts `PipelineStageInfo` and returns the `nn.Module` for that stage. This ensures construction consistency.

```python
from torch import Tensor
import torch.nn.functional as F

from d9d.core.dist_context import DistributedContext
from d9d.pipelining.factory import build_schedule, PipelineSchedule1F1BConfig


# 0. Define an object that manages loss calculation per microbatch. It receives the PipelineOutput
#    produced by the last stage and reads its fields by attribute.
class MyLossHandler:
    def __init__(self, targets_microbatches: list[Tensor]):
        self._targets = targets_microbatches

    def compute_loss(self, outputs: MyOutput, microbatch_idx: int):
        # Implement any custom logic here
        current_target = self._targets[microbatch_idx]
        return F.cross_entropy(outputs.logits.view(-1, outputs.logits.shape[-1]), current_target.view(-1))


# 1. Define configuration
dist_context: DistributedContext = ...
model_config = ...
schedule_config = PipelineSchedule1F1BConfig(
    num_stages_per_rank=4,  # 4 Virtual stages per rank
    zero_bubble=True  # Enable ZB1P optimization
)

# 2. Build the per-microbatch inputs (the "pack"). The number of microbatches is decided here, per
#    step. Each stage receives one PipelineInput and one SharedInput per microbatch.
inputs_microbatches = tuple(MyInput(input_ids=mb) for mb in my_input_microbatches)
shared_microbatches = tuple(MyShared(position_ids=pos) for pos in my_position_microbatches)
targets_microbatches = [...]  # one target tensor per microbatch

# 3. Build the schedule and model shards (the callback is supplied per step, not here)
loss_handler = MyLossHandler(targets_microbatches)
schedule_info, modules = build_schedule(
    dist_context=dist_context,
    schedule_config=schedule_config,
    model_provider=lambda stage: MyModelChunk(stage, model_config),  # Factory function
)

# 4. Execution
# The callback is passed to each step, so it can close over per-step data (here, the targets).
schedule_info.schedule.step(
    inputs_microbatches=inputs_microbatches,
    shared_microbatches=shared_microbatches,
    callback=loss_handler.compute_loss,
)
```

::: d9d.pipelining.api

::: d9d.pipelining.factory
