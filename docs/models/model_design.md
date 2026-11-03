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

Instead, d9d composes each architecture explicitly, as Hugging Face Transformers does. The call stack of each model is distinct, so its logic is easy to trace.

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

A segment is a `StageSegment`: a name, the modules that hold its parameters and a function `call(activations, constants)`. A stage module that implements `ModuleSupportsSegments` has two methods:

*   **`prepare_segment_constants(shared)`**: Builds the values that every segment of a microbatch receives and that need no gradient, e.g. RoPE embeddings.
*   **`stage_segments()`**: Returns the segments of the stage in forward order.

Each segment gets the activations that the previous one returned, and the first segment gets the stage input. The output of the last segment is the stage output. Use `run_segments()` as `forward`, so the model runs the same code with and without the runtime.

### Example

```python
from functools import partial

from d9d.pipelining.api import ModuleSupportsPipelining, StageSegment, run_segments


@dataclasses.dataclass
class MyConstants:
    rope: tuple[torch.Tensor, torch.Tensor]


@dataclasses.dataclass
class MyActivations:
    hidden_states: torch.Tensor


def _embed(inputs: MyInput, constants: MyConstants, *, embedding: nn.Module) -> MyActivations:
    return MyActivations(embedding(inputs.input_ids))


def _layer(activations: MyActivations, constants: MyConstants, *, layer: nn.Module) -> MyActivations:
    return MyActivations(layer(activations.hidden_states, constants.rope))


def _head(activations: MyActivations, constants: MyConstants, *, head: nn.Module) -> torch.Tensor:
    return head(activations.hidden_states)


class MyModel(nn.Module, ModuleSupportsPipelining, ModuleLateInit):
    def prepare_segment_constants(self, shared: MyShared) -> MyConstants:
        return MyConstants(rope=self.rope(shared.position_ids))

    def stage_segments(self) -> list[StageSegment]:
        segments = []
        if self._stage.is_current_stage_first:
            segments.append(StageSegment("embed", (self.embed,), partial(_embed, embedding=self.embed)))
        for name, layer in self.layers.items():
            segments.append(StageSegment(f"layer.{name}", (layer,), partial(_layer, layer=layer)))
        if self._stage.is_current_stage_last:
            segments.append(StageSegment("head", (self.head,), partial(_head, head=self.head)))
        return segments

    def forward(self, inputs: MyInput, shared: MyShared) -> torch.Tensor:
        return run_segments(self, inputs, shared)
```

`build_sequence_segments()` in `d9d.models.segments` builds these segments for a decoder with embedding, layers and norm, and the Qwen3 models use it.

### Rules

The runtime cuts the autograd graph between segments, so a segment sees only its arguments:

*   **Pass tensors with gradients through the activations**: Constants must not hold tensors with gradients, and the runtime rejects them. A tensor that a segment stores on `self` loses its gradient path to later segments, and nothing detects it.
*   **Keep no state between segments**: A module that collects values across layers, such as a hidden states aggregator, returns them instead, and the activations carry them.
*   **Cover every parameter**: Every parameter must belong to the modules of a segment, including frozen ones. The runtime checks this before the first forward pass.
*   **Keep the list fixed**: `stage_segments()` returns the same segments on every call. The runtime plans its buffers once.
*   **Keep tied parameters on one stage**: A parameter that several segments share, such as an embedding that the LM head reuses, works within one stage.
*   **Compute with plain tensors**: Inside a segment, the parameters are plain tensors, not `DTensor`s.

A segment is the unit that the runtime gathers and frees. One segment per decoder layer matches FSDP per layer. Smaller segments lower the peak memory but make smaller collectives, which are slower between nodes. A segment without parameters, such as one that only repacks the activations, costs nothing.

## Reference Implementations

See [Qwen3 MoE](./qwen3_moe.md) for a reference implementation.

## API Reference

::: d9d.nn
