# Pipeline Parallelism

## About

Pipeline parallelism splits a model into stages and places them on the ranks of the `pp` mesh dimension. Each rank builds only the stages it hosts. A schedule runs the microbatches of a step through the stages and exchanges stage outputs and gradients over P2P communication. d9d supports several schedules, from GPipe to DualPipeV, and runs them all on one execution engine.

## Shape Inference

A P2P receiver must know the shape of the incoming tensor to allocate its buffer. d9d asks your model to compute these shapes with a small protocol (`ModuleSupportsPipelining`). The model derives the shapes from the input arithmetically. d9d does not run a forward pass or trace the graph to find them.

Shapes can therefore change between steps, for example when the sequence length varies.

## Construction Consistency

A common pattern in distributed training is "instantiate, then delete": build the full model on a CPU or meta device, then cut it apart with `del model.layers[N:]`.

d9d rejects this pattern for three reasons:

1.  **Fragility**: A change to the model architecture requires a change to the external slicing script.
2.  **Leaky abstractions**: Forward methods fill up with checks like `if self.layer is not None`.
3.  **Invalid states**: The model object is half-built until it is sliced.

In d9d, models are **pipeline-aware**. Each pipeline rank builds only the stages it owns. The returned module is complete and valid right away.

## Making Models Compatible

### The Four I/O Roles

A pipelined model moves data across stage boundaries as four explicitly named, generic [PyTree](../pytree/index.md) types. Dataclasses are the recommended form.

| Role             | Meaning                                                     | Crosses P2P? |
|------------------|-------------------------------------------------------------|--------------|
| `PipelineInput`  | Input to the **first** stage (built by the task)            | no           |
| `StageTransfer`  | Payload between adjacent stages (out of *N* == in of *N+1*) | **yes**      |
| `PipelineOutput` | Output of the **last** stage (to the step handler)         | no           |
| `SharedInput`    | Value passed to **every** stage, rebuilt locally per rank   | no           |

Only `StageTransfer` crosses the network, so it is the only role that needs a `TensorSpec`.

### The Protocol

To use pipeline parallelism, your model implements `d9d.pipelining.api.ModuleSupportsPipelining[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]`:

*   **`forward(inputs, shared)`**: `inputs` is the `PipelineInput` on the first stage and the incoming `StageTransfer` on the other stages. It returns the outgoing `StageTransfer` on non-last stages and the `PipelineOutput` on the last stage. The stage knows its position from the `PipelineStageInfo` it gets at construction. It branches on `is_current_stage_first` and `is_current_stage_last` explicitly.
*   **`stage_transfer_spec(pipeline_input, boundary)`**: Returns a PyTree with the same structure as `StageTransfer`, with every tensor leaf replaced by a `TensorSpec`. The `boundary` (`StageBoundary.incoming` or `outgoing`) selects which stage edge to describe.

The `outgoing` transfer of stage *N* and the `incoming` transfer of stage *N+1* have the same dataclass type. So `pytree.tree_flatten` gives the same leaf order on both ends. Sender and receiver agree on the order by construction, and they need no handshake.

### Resumable Forwards

`ModuleSupportsWindows` describes a forward that can suspend at explicit execution boundaries. Implement `forward_steps(steps, inputs, shared)` and delegate synchronous `forward()` to `run_steps()`:

```python
import torch
from torch import nn

from d9d.pipelining.api import WindowSteps, mark_window, run_steps


class ResumableMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.first = mark_window(nn.Sequential(nn.Linear(width, width), nn.Tanh()))
        self.second = mark_window(nn.Linear(width, width))

    async def forward_steps(self, steps: WindowSteps, inputs: torch.Tensor, shared: None) -> torch.Tensor:
        hidden = await steps.call(self.first, inputs)
        return await steps.call(self.second, hidden)

    def forward(self, inputs: torch.Tensor, shared: None) -> torch.Tensor:
        return run_steps(self, inputs, shared)
```

`mark_window()` declares a parameter group without wrapping the module or changing its checkpoint names. The stage root owns parameters outside marked descendants. `call()` requests an ordinary module's group before invoking it and finishes the region with its output. For a resumable module, it composes that module's `forward_steps()` and finishes its final region. In eager execution, nested modules use their normal calls and hooks. Differentiable operations stay inside these module calls. For arbitrary computation between boundaries, use the manual form:

```python
async def forward_steps(self, steps: WindowSteps, inputs: torch.Tensor, shared: None) -> torch.Tensor:
    inputs = await steps.enter(self.first, inputs)
    hidden = self.first(inputs)
    hidden, skip = await steps.enter(self.second, (hidden, inputs))
    return self.second(hidden) + skip
```

Manual boundaries expose every new differentiable value from the open region that later regions need, including residuals. The caller reassigns the returned state. Previously returned boundary tensors retain their producers and can be saved for later calls. Before switching from a manual region to `call()`, use `state = await steps.finish(state)`. A helper coroutine's return does not finish its region; only the executor's outer coroutine has that implicit finish.

Each invocation receives a separate `WindowSteps`. Executors advance the coroutine to `WindowBoundary` events and resume it with the processed state. A boundary's `module` names the next parameter group; `None` finishes the current region without entering another one. User contexts such as `no_grad()` and autocast must not span suspension points, and the window executor raises an error if one does. Executors own compute contexts around each resume and coroutine cleanup. This protocol does not use an `asyncio` event loop.

`run_steps()` immediately resumes boundaries without graph cuts or parameter management. The parameter runtime instead gathers groups on demand and interleaves microbatches at boundaries. The first pass learns the group order; later passes prefetch from that order. An explicit `prefetch=next_module` on `enter()` names the next group and must match the next request.

Every forward while the runtime is bound must use the same group order, including repeated groups. Microbatches in a window and participating ranks must agree. A model without `forward_steps()` executes as one whole-stage group. Conditional token routing inside a group is supported; changing the group sequence is not.

### Example

Below is a skeleton of a Transformer-like model that supports d9d pipelining.

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
class MyShared:             # SharedInput (passed to every stage)
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

        # 1. Determine which layers live here.
        self.start_layer, self.end_layer = distribute_layers_for_pipeline_stage(
            config.n_layers, num_virtual_layers_pre=1, num_virtual_layers_post=1, stage=stage
        )

        # 2. Build sub-modules (ModuleDict keys keep the global layer index in parameter names).
        self.layers = nn.ModuleDict({
            str(layer): TransformerBlock(...)
            for layer in range(self.start_layer, self.end_layer)
        })

        # Only build embeddings on the first stage.
        if stage.is_current_stage_first:
            self.embed = nn.Embedding(...)

        # Only build the head on the last stage.
        if stage.is_current_stage_last:
            self.head = nn.Linear(...)

    def forward(self, inputs: MyInput | MyTransfer, shared: MyShared) -> MyTransfer | MyOutput:
        # Branch on the stage position, never on which argument is None.
        if self.stage.is_current_stage_first:
            x = self.embed(inputs.input_ids)
        else:
            x = inputs.hidden_states

        # Run local layers.
        for layer_idx in range(self.start_layer, self.end_layer):
            x = self.layers[str(layer_idx)](x)

        # The last stage produces the PipelineOutput; every other stage produces a StageTransfer.
        if self.stage.is_current_stage_last:
            return MyOutput(logits=self.head(x))
        return MyTransfer(hidden_states=x)

    # --- Protocol Implementation ---
    # Receives one microbatch of PipelineInput (not a global batch), so shapes are used as they are.
    # Returns a StageTransfer of TensorSpec descriptors and allocates no tensors.
    # The engine never calls this for the `incoming` edge of the first stage or the `outgoing` edge
    # of the last stage, so these edges need no special case.
    def stage_transfer_spec(self, pipeline_input: MyInput, boundary: StageBoundary) -> MyTransfer:
        batch_size, seq_len = pipeline_input.input_ids.shape
        return MyTransfer(
            hidden_states=TensorSpec(
                shape=(batch_size, seq_len, self.config.hidden_size), dtype=torch.bfloat16
            )
        )
```

## Supported Schedules

| Example JSON                                                          | Description                                                                                                                                                                                                       |
|:----------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `{"schedule": "inference"}`                                           | Inference only. Runs all forward passes and no backward passes.                                                                                                                                                   |
| `{"schedule": "gpipe"}`                                               | [GPipe](https://arxiv.org/abs/1811.06965). Hosts one stage per rank. Runs the forward pass of all microbatches before the backward pass.                                                                          |
| `{"schedule": "looped_bfs", "num_stages_per_rank": 2}`                | [Looped Breadth-First](https://arxiv.org/abs/2211.05953). Hosts several stages per rank. Runs all work for one stage before it moves to the next.                                                                 |
| `{"schedule": "1f1b", "num_stages_per_rank": 1, "zero_bubble": true}` | [Interleaved 1F1B](https://arxiv.org/abs/2104.04473), or [Interleaved Zero Bubble](https://arxiv.org/abs/2401.10241) with `zero_bubble`. Hosts several stages per rank. Zero Bubble splits the backward pass into input-gradient and weight-gradient parts. |
| `{"schedule": "zero_bubble_v"}`                                       | [Zero Bubble V](https://arxiv.org/abs/2401.10241). Hosts 2 stages per rank in a V shape. Splits the backward pass into input-gradient and weight-gradient parts.                                                  |
| `{"schedule": "dual_pipe_v"}`                                         | [DualPipeV](https://github.com/deepseek-ai/DualPipe). A bidirectional schedule that hosts 2 stages per rank in a V shape and pairs forward and backward passes of different microbatches.                         |

Some schedules limit the number of microbatches per step:

*   `dual_pipe_v` needs at least `2 * pp_size` microbatches.
*   `1f1b` needs a microbatch count that is divisible by `max(1, num_microbatches // pp_size)`. A multiple of `pp_size` always works.

## Microbatches and Packs

Pipelining consumes a **pack**: a sequence of ready microbatches for one step. The pack length, and so the batch size, can change from step to step. Each stage infers buffer shapes per microbatch, so the microbatches in one pack can also differ in shape.

The schedule composes its program once per microbatch count and reuses it. It reallocates buffers when a microbatch shape differs from the previous step.

## Usage

### Within the Trainer

Pipelining is available in the [`Trainer`](../loop/train.md) framework. Set the schedule in the `pipelining` section of the `Trainer` config:

```yaml
pipelining:
  schedule:
    schedule: 1f1b
    num_stages_per_rank: 2
    zero_bubble: true
```

The `Trainer` builds the schedule and distributes the layers.

### Manual Usage

To use pipelining outside the `Trainer`, for example in a custom loop, call the `build_schedule` factory.

`build_schedule` takes a model provider instead of a built model. The model provider is a function that accepts a `PipelineStageInfo` and returns the `nn.Module` for that stage. This keeps construction consistent.

`build_schedule` also takes the program builder of the schedule. `build_program_builder` builds it from a schedule config. You can also pass your own builder (see [Building Custom Schedules](./internals.md#building-custom-schedules)).

```python
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import DeviceMesh

from d9d.pipelining.api import PipelineStepHandler
from d9d.pipelining.factory import PipelineSchedule1F1BConfig, build_program_builder, build_schedule


# 0. Define the step handler. It computes the loss per microbatch from the PipelineOutput of the
#    last stage. It can also override compute_context() to open a reusable context around each
#    forward and backward pass of a stage.
class MyStepHandler(PipelineStepHandler[MyOutput]):
    def __init__(self, targets_microbatches: list[Tensor]):
        self._targets = targets_microbatches

    def process_outputs(self, outputs: MyOutput, microbatch_idx: int) -> Tensor:
        # Implement any custom logic here.
        current_target = self._targets[microbatch_idx]
        return F.cross_entropy(outputs.logits.view(-1, outputs.logits.shape[-1]), current_target.view(-1))


# 1. Define the configuration.
pp_mesh: DeviceMesh = ...  # The pipeline-parallel mesh, e.g. dist_context.mesh_for(BASE_DOMAIN)[BaseDim.PP]
model_config = ...
schedule_config = PipelineSchedule1F1BConfig(
    num_stages_per_rank=4,  # 4 virtual stages per rank
    zero_bubble=True  # Use the ZB1P variant
)

# 2. Build the per-microbatch inputs (the pack). The number of microbatches is decided here, per
#    step. Each stage receives one PipelineInput and one SharedInput per microbatch.
inputs_microbatches = tuple(MyInput(input_ids=mb) for mb in my_input_microbatches)
shared_microbatches = tuple(MyShared(position_ids=pos) for pos in my_position_microbatches)
targets_microbatches = [...]  # One target tensor per microbatch

# 3. Build the schedule and the model stages (the handler is passed per step, not here).
#    `modules` maps the global index of each stage on this rank to its module.
step_handler = MyStepHandler(targets_microbatches)
schedule_info, modules = build_schedule(
    pp_mesh=pp_mesh,
    program_builder=build_program_builder(schedule_config),
    model_provider=lambda stage: MyModelChunk(stage, model_config),  # Factory function
)

# 4. Run the step.
# The handler is passed to each step, so it can use per-step data (here, the targets).
schedule_info.schedule.step(
    inputs_microbatches=inputs_microbatches,
    shared_microbatches=shared_microbatches,
    handler=step_handler,
)
```

## API Reference

::: d9d.pipelining.api

::: d9d.pipelining.factory
