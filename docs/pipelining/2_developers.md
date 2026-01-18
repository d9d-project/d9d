---
title: Internals
---

# Internals

This section details the internals of the `d9d.pipelining` module. It is intended for those who wish to implement new layouts, schedules, or modify the execution engine.

## Architecture

### The Idea

d9d decouples the **Schedule Structure** from the **Runtime Execution**.

1.  You write a builder (e.g., `1F1B`, `DualPipe`) that generates a linear list of logical `Actions` (e.g., `Forward(Stage=0, MB=1)`, `Backward(Stage=0, MB=0)`). If you want, d9d may automatically inject `Send`/`Recv` actions into your compute-only schedule based on data dependencies, preventing deadlocks.
2.  You run a dumb virtual machine simply iterates the action list and executes them.

This makes implementing complex research schedules (like Zero Bubble or DualPipeV) significantly easier than managing state machines or recursive calls.

### Core Components

#### PipelineStage (`infra/stage/stage.py`)

Encapsulates a user `nn.Module`. It is **not** responsible for deciding *when* to run. Instead, it provides atomic pipeline stage capabilities (such as forward and backward passes) to the actions and the executor.

Consists of:

*   **Computation Handlers**: 
    *   `ForwardComputeHandler`: Performs forward pass, caches inputs/outputs for backward passes.
    *   `BackwardComputeHandler`: Performs backward pass, capable of splitting backward passes into `backward_input` (dI) and `backward_weight` (dW) for advanced schedules.
*   **Communication Handlers**: Contain and manage the P2P buffers for both forward and backward passes.

#### Actions (`infra/schedule/component/runtime/action.py`)

The atomic instructions for the pipeline virtual machine.

*   `ForwardComputeAction`: Run forward on specific microbatch.
*   `BackwardFullInputComputeAction`: Run backward. Can be configured to compute gradients for inputs-only or inputs+weights.
*   `BackwardWeightComputeAction`: Compute gradients for weights (used in Zero Bubble schedules).
*   `ForwardSendAction` / `ForwardReceiveAction` / `BackwardSendAction` / `BackwardReceiveAction`: Network IO.
*   `ComposeAction`: Composes multiple actions into a single one. Used for Forward/Backward overlap in schedules such as DualPipeV.

Actions are designed to be declarative and immutable.

#### Programs

A `Program` is simply `dict[int, list[ActionBase]]` — a mapping of Rank ID to a sequential list of Actions.

#### Executor (`infra/schedule/component/runtime/executor.py`)

The `PipelineScheduleExecutor` is the runtime engine. 

It:

1.  Shards global inputs into microbatches.
2.  Iterates through the `Program` action list.
3.  Dispatches calls to `Action`s that perform computation or communication workload.

### Comparison with PyTorch

The d9d pipelining implementation is heavily inspired by and borrows concepts from the `torch.distributed.pipelining` API (e.g., ZeroBubble implementation), but refactors the codebase significantly to improve clarity, type safety, and modularity.

The main architectural differences lie in the **strict separation of concerns** and **composition over inheritance**:

1.  **Decomposed Stage Logic**:
    *   **PyTorch**: Uses a monolithic `_PipelineStageBase` class that simultaneously manages P2P buffer allocation, gradient accumulation state, and forward/backward execution logic.
    *   **d9d**: Adopts a compositional approach. The `PipelineStage` class is a thin orchestrator that delegates responsibilities to dedicated handlers.

2.  **Polymorphic Actions vs Enumeration**:
    *   **PyTorch**: Represents schedule instructions using a single generic `_Action` NamedTuple combined with an Enum (`_ComputationType.FORWARD`, `_ComputationType.SEND_F`, etc.).
    *   **d9d**: Uses a class hierarchy for actions (`ForwardComputeAction`, `ForwardSendAction`, `ComposeAction`). This allows the runtime executor to use structural pattern matching (`match/case`) rather than large `if/elif` blocks checking enums, allows different actions to carry different metadata (e.g. `full_backward` flag), and improves static type checking.

3.  **Builder Pattern vs Schedule Classes**:
    *   **PyTorch**: Often couples the schedule definition with the runtime object (e.g., `Schedule1F1B` class contains both the logic to generate the ordering and the logic to execute it).
    *   **d9d**: Strictly separates the **Program Builder** (which generates the list of actions) from the **Executor** (which runs the actions). This makes it easier to inspect a schedule plan before execution or swap scheduling algorithms without changing the runtime driver.

## Building Custom Schedules

To build a new schedule, you create a `PipelineProgramBuilder`.

### Implement the Builder

You must implement the pipeline program builder.

```python
from collections import defaultdict

from d9d.pipelining.infra.schedule.component.program import PipelineProgramBuilder, build_stage_to_host_rank_topology, ScheduleStyle, add_communication_ops
from d9d.pipelining.infra.schedule.component.runtime import ActionBase, ForwardComputeAction


class MyFancyScheduleBuilder(PipelineProgramBuilder):
    def __init__(self, stages_per_rank: int):
        self._stages_per_rank = stages_per_rank

    @property
    def num_stages_per_rank(self) -> int:
        return self._stages_per_rank

    @property
    def topology_style(self) -> ScheduleStyle:
        return ScheduleStyle.loop

    def compose(self, num_microbatches: int, pp_size: int) -> dict[int, list[ActionBase]]:
        # Map logical stages to ranks
        stage_to_rank = build_stage_to_host_rank_topology(num_stages=self._stages_per_rank * pp_size,
                                                          style=ScheduleStyle.loop,
                                                          pp_size=pp_size)

        actions = defaultdict(list)

        # 1. Generate Compute Schedule
        for rank in range(pp_size):
            # ... custom logic to decide order of Fwd/Bwd ...
            actions[rank].append(ForwardComputeAction(stage_idx=..., microbatch_idx=...))

        # 2. Inject Communications (Magic Pass)
        # This analyzes data dependencies between stages and inserts Send/Recvs
        return add_communication_ops(actions, stage_to_rank, num_stages=self._stages_per_rank * pp_size)
```

### Registering

Add your configuration to `factory/config.py` and register the builder in `factory/factory.py`.

::: d9d.pipelining.infra.stage
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.pipelining.infra.schedule.component.runtime
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.pipelining.infra.schedule.component.program
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.pipelining.infra.schedule.program
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.pipelining.training
    options:
        show_root_heading: true
        show_root_full_path: true
