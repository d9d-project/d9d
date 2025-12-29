---
title: Model State Mapper
---

# Model State Mapper

## About

The `d9d.model_state.mapper` package solves the complexity of working with model checkpoints by providing 
**a declarative, graph-based framework for transforming model states**.

## Core Concept

Loading large-scale models is rarely a simple 1-to-1 key matching operation. You often face challenges such as:

*   **Naming Mismatches**: HuggingFace uses `model.layers.0`, your custom model uses `transformer.h.0`.
*   **Shape Mismatches**: The checkpoint stores `Q`, `K`, and `V` separately, but your model implementation expects a stacked `QKV` tensor.
*   **Scale**: The checkpoint is 500GB. You cannot load the whole dictionary on every GPU to process it.

Instead of writing a manual loop that loads tensors and blindly modifies them, this framework treats state transformation as a **Directed Acyclic Graph (DAG)**.

Such a declarative approach makes it available for d9d to perform complex transform-save and transform-load operations 
effectively in a streamed manner without loading the whole checkpoint into memory.

## Usage Examples

### Pass-through Mapping for PyTorch Module

If you simply want to load a checkpoint where keys match the model definition (standard load_state_dict behavior), but 
want to utilize d9d's streaming/sharding capabilities.

```python
import torch.nn as nn
from d9d.model_state.mapper.adapters import identity_mapper_from_module

# Define your PyTorch model
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 5)
)

# Automatically generate a mapper based on the model's actual parameter names
# This creates Identity mappers for "0.weight", "0.bias", "2.weight", "2.bias"
mapper = identity_mapper_from_module(model)
```

### Using Leaf Mappers

This example demonstrates using leaf mappers to handle common mismatch scenario: merging separate Query/Key/Value 
tensors into a single tensor.

```python
import torch
from d9d.model_state.mapper.leaf import (
    ModelStateMapperRename,
    ModelStateMapperStackTensors
)

# Stacking Tensors
# Scenario: Checkpoint has separate Q, K, V linear layers, we need one QKV tensor
stack_mapper = ModelStateMapperStackTensors(
    source_names=["attn.q.weight", "attn.k.weight", "attn.v.weight"],
    target_name="attn.qkv.weight",
    stack_dim=0
)

# To show what this mapper needs:
print(stack_mapper.state_dependency_groups())
# Output: {StateGroup(inputs={'attn.q.weight', ...}, outputs={'attn.qkv.weight'})}

# To actually execute:
dummy_data = {
    "attn.q.weight": torch.randn(64, 64),
    "attn.k.weight": torch.randn(64, 64),
    "attn.v.weight": torch.randn(64, 64),
}
result = stack_mapper.apply(dummy_data)
print(result["attn.qkv.weight"].shape) 
# Output: torch.Size([3, 64, 64])
```

### Composing Complex Pipelines

Converting an entire model state requires processing multiple keys in parallel, and potentially chaining 
operations (e.g., Rename then Stack).

```python
from d9d.model_state.mapper.compose import ModelStateMapperSequential, ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperRename, ModelStateMapperStackTensors

# Define a transformation pipeline
mapper = ModelStateMapperSequential([
    # Step 1: Rename keys to standard format
    ModelStateMapperParallel([
        ModelStateMapperRename("bert.encoder.layer.0.attention.self.query.weight", "layer.0.q"),
        ModelStateMapperRename("bert.encoder.layer.0.attention.self.key.weight", "layer.0.k"),
        ModelStateMapperRename("bert.encoder.layer.0.attention.self.value.weight", "layer.0.v"),
    ]),
    # Step 2: Stack them into a specialized attention tensor
    ModelStateMapperStackTensors(
        source_names=["layer.0.q", "layer.0.k", "layer.0.v"],
        target_name="layer.0.qkv",
        stack_dim=0
    )
])
```

::: d9d.model_state.mapper
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.model_state.mapper.adapters
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.model_state.mapper.compose
    options:
        show_root_heading: true
        show_root_full_path: true

::: d9d.model_state.mapper.leaf
    options:
        show_root_heading: true
        show_root_full_path: true
