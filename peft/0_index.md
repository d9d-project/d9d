---
title: PEFT Overview
---

# Parameter-Efficient Fine-Tuning (PEFT)

## About

The `d9d.peft` package provides a flexible framework for fine-tuning models using parameter-efficient strategies or targeted full fine-tuning.

## Core Concepts

### Apply Before State Loading

This package is deeply integrated with the [model state mapping](../model_states/mapper.md) ecosystem. 

When you apply methods like LoRA, the model structure changes (e.g., a `Linear` layer becomes a `LoRALinear` wrapper). 

Consequently, the keys in your original checkpoint (e.g., `layers.0.linear.weight`) no longer match the keys in the efficient model (e.g., `layers.0.linear.base.weight`). 

`d9d.peft` automatically generates the necessary `ModelStateMapper` objects to load standard checkpoints into modified architectures. 

It is useful since framework user may apply a PEFT method to a model that was not initialized or [horizontally distributed](../models/2_horizontal_parallelism.md) yet. 
Other PEFT frameworks usually want you to initialize model weights **before** applying PEFT which may break your horizontal parallelism setup logic or make it less reusable.


### Configuration

All PEFT methods are driven by Pydantic configurations. This allows for custom validation of hyperparameters and easy serialization/deserialization.

### The Injection Lifecycle (`PeftMethod`)

The framework operates on an **Inject -> Train -> Merge** lifecycle:

1.  **Inject** (`inject_peft_and_freeze`): The `PeftMethod` inspects the generic `nn.Module`. It locates target layers, replaces them with adapter layers (if necessary), and marks parameters that have to be trained with `requires_grad=True`.
2.  **State Mapping**: The injection process returns a `ModelStateMapper` object. This mapper describe how to map the *original* checkpoint keys to the *new, injected* model structure.
3.  **Train**: Here you train your model.
4.  **Merge** (`merge_peft`): Once training is complete, this method collapses the adapters back into the base weights, restoring the original architecture.

::: d9d.peft
    options:
        show_root_heading: true
        show_root_full_path: true
