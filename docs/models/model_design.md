# Model Design

## About

d9d does not require you to use its model implementations. You can bring your own model and optionally build it from d9d blocks. Your model must follow the principles on this page to work with the d9d training loop.

## Principles

d9d uses a "white-box" approach to modeling. Models are plain, readable PyTorch code without heavy abstraction layers.

### No Layer Specs

Some distributed frameworks make you describe a model with metadata objects. The framework then injects wrapping logic, such as FSDP or activation checkpointing. This makes debugging difficult.

In d9d, you write standard `nn.Module` classes. You can use `nn.Linear`, `nn.RMSNorm` or d9d blocks directly. You apply distributed strategies to submodules after construction, so the model code stays standard PyTorch.

### Explicit Composition

d9d avoids "uber-modules": large classes such as a `GenericTransformerBlock` that cover every architecture variant through dozens of flags. Examples of such variants are MoE vs. dense, pre-norm vs. post-norm and parallel attention.

Instead, d9d composes each architecture explicitly, as Hugging Face Transformers does. Each model builds its own layers, so its logic is easy to trace. Models with the same chain of layers, such as Qwen3 Dense and Qwen3 MoE, share the code that describes this chain as [segments](../pipelining/index.md#segments).

### Pipelining-Aware Models

See [Pipeline Parallelism](../pipelining/index.md).

### Late Initialization

Building a large model on a single GPU, or even in CPU RAM, often runs out of memory. d9d avoids this with the `ModuleLateInit` protocol. Every model stage that you pass to the [`Trainer`](../loop/train.md) must implement it.

The `Trainer` initializes a model stage in this order:

1.  Construct the model on the `meta` device, which allocates no memory.
2.  Place the parameters with the horizontal parallelism strategy.
3.  Allocate empty storage for the local shards on the target device.
4.  Call `reset_parameters()` to initialize the weights.
5.  Load the source checkpoint, if one is configured.

## Segments

The [Parameter Runtime](../parallel/runtime.md) gathers the weights of a model part by part. A model tells it the parts by describing its pipeline stage as a chain of segments. A model without segments still works: it runs as one segment, and FSDP then holds all weights of the stage at once.

### The Protocol

A segment is a `StageSegment`: a name and an `nn.Module` that owns its parameters and computes its activations. A stage module that implements `ModuleSupportsSegments` has two methods:

*   **`prepare_segment_constants(shared)`**: Builds the values that every segment of a microbatch receives and that need no gradient, e.g. RoPE embeddings.
*   **`stage_segments()`**: Returns the segments of the stage in forward order.

Each segment gets the activations that the previous one returned, and the first segment gets the stage input. The output of the last segment is the stage output. Use `run_segments()` as `forward`, so the model runs the same code with and without the runtime.

### Example

```python
import torch
from torch import nn

from d9d.pipelining.api import ModuleSupportsSegments, StageSegment, run_segments


class LayerSegment(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self._layer = layer

    def forward(self, activations: torch.Tensor, constants: None) -> torch.Tensor:
        return torch.relu(self._layer(activations))


class SegmentedMLP(nn.Module, ModuleSupportsSegments):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(8, 16)
        self.down = nn.Linear(16, 8)

    def prepare_segment_constants(self, shared: None) -> None:
        return shared

    def stage_segments(self) -> list[StageSegment]:
        return [StageSegment("up", LayerSegment(self.up)), StageSegment("down", LayerSegment(self.down))]

    def forward(self, inputs: torch.Tensor, shared: None) -> torch.Tensor:
        return run_segments(self, inputs, shared)


model = SegmentedMLP()
output = model(torch.ones(2, 8), None)
```

The segment modules wrap the layers and are not children of the model, so checkpoint names remain `up.*` and `down.*`. The Qwen3 models use `build_sequence_segments()` from `d9d.models.segments` for their embedding, layers and norm.

### Rules

The runtime cuts the autograd graph between segments, so a segment sees only its arguments:

*   **Pass tensors with gradients through the activations**: Constants must not hold tensors with gradients, and the runtime rejects them. A tensor that a segment stores on `self` loses its gradient path to later segments, and nothing detects it.
*   **Keep no state between segments**: A module that collects values across layers, such as a hidden states aggregator, returns them instead, and the activations carry them.
*   **Cover every parameter**: Every parameter must belong to a segment module, including frozen ones. The runtime checks this before the first forward pass.
*   **Keep the list fixed**: After materialization, segment names, order and parameter identities stay the same. The runtime reads the segments once per bind.
*   **Keep tied parameters on one stage**: A parameter that several segments share, such as an embedding that the LM head reuses, works within one stage.
*   **Compute with plain tensors**: Inside a segment, the parameters are plain tensors, not `DTensor`s.

A segment is the unit that the runtime gathers and frees. One segment per decoder layer matches FSDP per layer. Smaller segments lower the peak memory but make smaller collectives, which are slower between nodes. A segment without parameters, such as one that only repacks the activations, costs nothing.

## Reference Implementations

See [Qwen3 MoE](./qwen3_moe.md) for a reference implementation.

## API Reference

::: d9d.nn
