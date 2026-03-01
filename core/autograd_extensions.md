---
title: Autograd Extensions
---

# Autograd Extensions

## About

The `d9d.core.autograd` package provides utilities to exert fine-grained control over PyTorch's automatic differentiation engine.


## The Global Grad Context

### Why

The primary purpose of so-called Global Grad Context is to solve specific limitations in `torch.autograd.Function` 
regarding partial backward passes, which are critical for advanced distributed training schedules like Zero-Bubble Pipeline Parallelism.

In standard PyTorch operations (like `torch.matmul`), the autograd engine is highly optimized. 
If you perform a backward pass specifying only a subset of inputs (e.g., `torch.autograd.backward(..., inputs=[activations])`), 
PyTorch will intelligently skip computing gradients for parameters (weights) to save compute.

However, custom `torch.autograd.Function` implementations **do not** share this intelligence. 
PyTorch sets `ctx.needs_input_grad` to `True` for every input that has `requires_grad=True`, regardless of whether 
that specific edge is actually being computed in the current `backward()` call.

This behavior makes it impossible to implement split-backward pipeline schedules (where activation gradients and weight 
gradients are computed at different times) using custom operations (like GroupedGEMM) without performing redundant 
calculations.

For more details, see [PyTorch Issue #174017](https://github.com/pytorch/pytorch/issues/174017).

### How it Works

To bypass this limitation, `d9d` introduces the `GlobalGradContext`. It acts as a side-channel state manager that 
allows the training loop to explicitly signal its intent to the custom operators.

1.  **Orchestrator**: The training loop sets the context (e.g., "I only want Input gradients now").
2.  **Operator**: The custom `backward` checks this context. Even if PyTorch says `needs_input_grad=True`, the operator will verify with `GlobalGradContext` before computation.

## Usage

### In Custom Autograd Functions

When writing a custom operation, you must tag your gradients with a semantic `GradDirection` and check the context before computation.

```python
import torch
from d9d.core.autograd import GLOBAL_GRAD_CONTEXT, GradDirection

class MyCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight):
        # Save which direction 'inputs' and 'weight' correspond to
        ctx.dir_inputs = GradDirection.inputs
        ctx.dir_weight = GradDirection.weight
        ctx.save_for_backward(inputs, weight)
        return torch.matmul(inputs, weight)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        
        # Check 1: Does PyTorch need it? AND Check 2: Does Context allow it?
        
        # Calculate Input Gradients (Activation)
        if ctx.needs_input_grad[0] and GLOBAL_GRAD_CONTEXT.check_direction(ctx.dir_inputs):
            grad_input = torch.matmul(grad_output, weight.t())

        # Calculate Weight Gradients
        if ctx.needs_input_grad[1] and GLOBAL_GRAD_CONTEXT.check_direction(ctx.dir_weight):
            grad_weight = torch.matmul(inputs.t(), grad_output)

        return grad_input, grad_weight
```

### In Training Loops

By default, the `GLOBAL_GRAD_CONTEXT` is set to compute both input and weight gradients. 

The d9d pipelining API configures it for split-backward automatically.
So, if you use the [Trainer](), **everything will work out of the box**.

If you use your own training loop implementation - you have to configure the context manually.

## API Reference

::: d9d.core.autograd
    options:
        show_root_heading: true
        show_root_full_path: true
