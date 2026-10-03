# PEFT Overview

## About

The `d9d.peft` package fine-tunes models with parameter-efficient methods, such as LoRA, or with full fine-tuning of selected modules.

## Apply Before State Loading

This package is built on the [model state mapping](../model_states/mapper.md) framework.

Methods like LoRA change the model structure. For example, an `nn.Linear` layer becomes a `LoRALinear` wrapper. The keys in the original checkpoint (e.g. `layers.0.linear.weight`) then no longer match the model keys (e.g. `layers.0.linear.base.weight`). `d9d.peft` returns a `ModelStateMapper` that renames only the keys the method changes. Chain it after your checkpoint mapper with `ModelStateMapperSequential`, so all other keys load unchanged.

So you can apply a PEFT method before the model is initialized or [horizontally distributed](../models/horizontal_parallelism.md). Other PEFT frameworks usually require initialized weights before they apply PEFT. That can break your horizontal parallelism setup or make it harder to reuse.

## Configuration

Every PEFT method has a Pydantic configuration. Pydantic validates the hyperparameters and serializes the configuration. `d9d.peft.all.peft_method_from_config` builds the method for any configuration (see [Method Stacking](./stack.md)).

## The Injection Lifecycle

Every PEFT method implements `PeftMethod` and follows an **inject, train, merge** lifecycle:

1.  **Inject** (`inject_peft_and_freeze`): The method finds the target layers in the `nn.Module` and replaces them with adapter layers if needed. All parameters are frozen except those the method trains.
2.  **State mapping**: The injection returns a `ModelStateMapper`. It maps the *original* keys of the changed layers to the *new* model structure. It does not cover the other keys.
3.  **Train**: You train the model.
4.  **Merge** (`merge_peft`): After training, the method merges the adapters into the base weights and restores the original architecture.

## Usage

See [LoRA](./lora.md), [Full Fine-Tuning](./full_tune.md) and [Method Stacking](./stack.md) for examples.

## API Reference

::: d9d.peft
