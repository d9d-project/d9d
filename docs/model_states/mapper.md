# Model State Mapper

## About

The `d9d.model_state.mapper` package is a declarative, graph-based framework for transforming model states. You describe which checkpoint keys map to which model keys, and d9d runs the transformation while it streams the checkpoint.

## Core Concept

Loading a large model is rarely a 1-to-1 key match. Common problems are:

*   **Naming mismatches**: Hugging Face uses `model.layers.0`, your model uses `transformer.h.0`.
*   **Shape mismatches**: The checkpoint stores `q`, `k` and `v` separately, but your model expects one stacked `qkv` tensor.
*   **Scale**: The checkpoint takes hundreds of GiB. You cannot load the whole dictionary on every GPU to process it.

A mapper does not loop over tensors and modify them. It treats the transformation as a directed acyclic graph (DAG). Each mapper declares its dependency groups: which input keys produce which output keys. d9d uses these groups to load, transform and save a checkpoint in a stream, without holding the whole checkpoint in memory.

Mappers come in three kinds:

*   **Leaf mappers** (`d9d.model_state.mapper.leaf`) transform individual tensors: rename, stack, chunk, transpose, distribute.
*   **Composite mappers** (`d9d.model_state.mapper.compose`) combine other mappers: in parallel, in sequence, under a key prefix or restricted to one shard.
*   **Adapters** (`d9d.model_state.mapper.adapters`) build identity mappers from a module or from another mapper.

## Usage

### Pass-Through Mapping for a PyTorch Module

Use `identity_mapper_from_module` when checkpoint keys match the model's state dict keys, as in standard `load_state_dict`. You still get d9d's streaming and sharding.

```python
import torch.nn as nn
from d9d.model_state.mapper.adapters import identity_mapper_from_module

model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 5)
)

# Creates identity mappers for "0.weight", "0.bias", "2.weight" and "2.bias".
mapper = identity_mapper_from_module(model)
```

### Leaf Mappers

This example merges separate query, key and value tensors into a single tensor.

```python
import torch
from d9d.model_state.mapper.leaf import ModelStateMapperStackTensors

stack_mapper = ModelStateMapperStackTensors(
    source_names=["attn.q.weight", "attn.k.weight", "attn.v.weight"],
    target_name="attn.qkv.weight",
    dim=0
)

# Show what this mapper needs and produces.
print(stack_mapper.state_dependency_groups())
# Output (abridged): {StateGroup(inputs={'attn.q.weight', ...}, outputs={'attn.qkv.weight'})}

# Run the transformation.
dummy_data = {
    "attn.q.weight": torch.randn(64, 64),
    "attn.k.weight": torch.randn(64, 64),
    "attn.v.weight": torch.randn(64, 64),
}
result = stack_mapper.apply(dummy_data)
print(result["attn.qkv.weight"].shape)
# Output: torch.Size([3, 64, 64])
```

### Composing Pipelines

A full model conversion processes many keys in parallel and can chain operations, e.g. rename, then stack.

```python
from d9d.model_state.mapper.compose import ModelStateMapperSequential, ModelStateMapperParallel
from d9d.model_state.mapper.leaf import ModelStateMapperRename, ModelStateMapperStackTensors

mapper = ModelStateMapperSequential([
    # Step 1: Rename keys to a short format.
    ModelStateMapperParallel([
        ModelStateMapperRename("bert.encoder.layer.0.attention.self.query.weight", "layer.0.q"),
        ModelStateMapperRename("bert.encoder.layer.0.attention.self.key.weight", "layer.0.k"),
        ModelStateMapperRename("bert.encoder.layer.0.attention.self.value.weight", "layer.0.v"),
    ]),
    # Step 2: Stack them into one attention tensor.
    ModelStateMapperStackTensors(
        source_names=["layer.0.q", "layer.0.k", "layer.0.v"],
        target_name="layer.0.qkv",
        dim=0
    )
])
```

## API Reference

::: d9d.model_state.mapper

::: d9d.model_state.mapper.adapters

::: d9d.model_state.mapper.compose

::: d9d.model_state.mapper.leaf
