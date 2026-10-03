# Autograd Extensions

## About

The `d9d.core.autograd` package gives fine-grained control over the PyTorch autograd engine. Its main tool is the global grad context. It tells custom autograd functions which gradients to compute in a partial backward pass, as split-backward pipeline schedules such as Zero Bubble need.

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
2.  **Operation**: The custom `backward` checks the context. It computes a gradient only if `needs_input_grad` is `True` **and** the context enables its direction.

By default, `GLOBAL_GRAD_CONTEXT` enables both input and weight gradients.

## Usage

### In Custom Autograd Functions

When you write a custom operation, assign a `GradDirection` to each gradient. Check the context before you compute it.

```python
import torch
from d9d.core.autograd import GLOBAL_GRAD_CONTEXT, GradDirection


class MyCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight):
        ctx.save_for_backward(inputs, weight)
        return torch.matmul(inputs, weight)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        # Compute a gradient only if PyTorch needs it AND the context enables its direction.
        if ctx.needs_input_grad[0] and GLOBAL_GRAD_CONTEXT.check_direction(GradDirection.inputs):
            grad_input = torch.matmul(grad_output, weight.t())

        if ctx.needs_input_grad[1] and GLOBAL_GRAD_CONTEXT.check_direction(GradDirection.weight):
            grad_weight = torch.matmul(inputs.t(), grad_output)

        return grad_input, grad_weight
```

### In Training Loops

The d9d pipelining schedules set the context for split-backward passes. If you use them, directly or through the [`Trainer`](../loop/train.md), you do not need to do anything.

If you write your own split-backward logic, you must set the context yourself:

```python
import torch
from d9d.core.autograd import GLOBAL_GRAD_CONTEXT, GradDirection

with GLOBAL_GRAD_CONTEXT.with_directions(GradDirection.inputs):
    torch.autograd.backward(outputs, grad_tensors=output_grads, inputs=stage_inputs, retain_graph=True)
```

## API Reference

::: d9d.core.autograd
