# Split Backward

## About

The `d9d.autograd.split_backward` package splits a backward pass into two phases: the gradients of the inputs and the gradients of the weights. Split-backward pipeline schedules, such as Zero Bubble, run the two phases at different times. The package has two parts. The split backward functions run the phases. The global grad context tells custom autograd functions which gradients the current phase needs.

## The Two Phases

A pipeline stage sends the gradients of its inputs to the previous stage. The weight gradients are needed only at the optimizer step. So a schedule computes the input gradients first and fills the idle time of the pipeline with the weight gradients later.

*   **`stage_backward_input()`**: Computes the gradients of the inputs and defers the weight gradients. It captures the gradients at the nodes of the autograd graph where the weight gradients branch off.
*   **`stage_backward_weight()`**: Replays the captured gradients and accumulates the weight gradients.
*   **`stage_backward_full()`**: Runs both phases in one backward pass.

The `PendingWeightBackward` from `stage_backward_input()` keeps the autograd graph alive. Keep it until you call `stage_backward_weight()`.

## The Global Grad Context

### Why

Split-backward pipeline schedules compute activation gradients and weight gradients at different times. They run a backward pass for a subset of tensors, e.g. `torch.autograd.backward(..., inputs=[activations])`.

For built-in operations such as `torch.matmul`, PyTorch then skips the weight gradients.

Custom `torch.autograd.Function` implementations do not get this behavior. PyTorch sets `ctx.needs_input_grad` to `True` for every input with `requires_grad=True`. It does so even if the current `backward()` call does not compute that edge.

So a custom operation, such as a grouped GEMM, computes all its gradients in every partial backward pass. This wastes compute in split-backward schedules.

For more details, see [PyTorch Issue #174017](https://github.com/pytorch/pytorch/issues/174017).

### How It Works

d9d adds the `GlobalGradContext` to work around this limitation. It is shared state through which the training loop tells custom operations which gradients it needs.

1.  **Training loop**: Sets the enabled gradient directions, e.g. "input gradients only".
2.  **Operation**: The custom `backward` calls `needs_input_grad()`. It computes a gradient only if autograd needs it and the context enables its direction.

By default, `GLOBAL_GRAD_CONTEXT` enables both input and weight gradients.

## Usage

### In Custom Autograd Functions

When you write a custom operation, assign a `GradDirection` to each input. Call `needs_input_grad()` instead of `ctx.needs_input_grad` before you compute a gradient.

```python
import torch
from d9d.autograd.split_backward import GradDirection, needs_input_grad


class MyCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight):
        ctx.save_for_backward(inputs, weight)
        return torch.matmul(inputs, weight)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        if needs_input_grad(ctx, 0, GradDirection.inputs):
            grad_input = torch.matmul(grad_output, weight.t())

        if needs_input_grad(ctx, 1, GradDirection.weight):
            grad_weight = torch.matmul(inputs.t(), grad_output)

        return grad_input, grad_weight
```

### In Training Loops

The d9d pipelining schedules run the split backward pass for you. If you use them, directly or through the [`Trainer`](../loop/train.md), you do not need to do anything.

To split the backward pass in your own schedule, call the two phases. They set the global grad context themselves:

```python
from d9d.autograd.split_backward import stage_backward_input, stage_backward_weight

# Phase 1: the input gradients. Send them to the previous stage, and keep `pending` until phase 2.
input_grads, pending = stage_backward_input(outputs, output_grads, inputs, weights=model.parameters())

# Phase 2, later in the schedule: the weight gradients.
stage_backward_weight(pending)
```

If you run the backward pass without these functions, set the context yourself:

```python
import torch
from d9d.autograd.split_backward import GLOBAL_GRAD_CONTEXT, GradDirection

with GLOBAL_GRAD_CONTEXT.with_directions(GradDirection.inputs):
    torch.autograd.backward(outputs, grad_tensors=output_grads, inputs=stage_inputs, retain_graph=True)
```

## API Reference

::: d9d.autograd.split_backward
