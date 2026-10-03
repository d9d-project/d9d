# PyTree Traversal

## About

The `d9d.core.pytree` package traverses nested tensor structures (pytrees) recursively. It is a thin wrapper around [`optree`](https://github.com/metaopt/optree) that also descends into dataclasses. d9d uses it wherever it applies an operation to every tensor in a nested structure. Examples are moving a microbatch to the device, moving metric results to the CPU and flattening a metric tree for logging.

## Supported Containers

The package traverses the containers that [`PyTree`](./types.md) describes (`dict`, `list`, `tuple`), nested to any depth. It also traverses any dataclass: the fields of a dataclass are its children in the tree.

Nesting works in every direction. A dataclass inside a dict, a list of dataclasses and a dataclass with dicts of tensors as fields are all traversed.

## Traversal Order

Traversal order is deterministic:

*   **`dict` keys** are traversed in sorted order, whatever the insertion order.
*   **Dataclass fields** are traversed in declaration order.

## Usage

Pass a dataclass, nested in containers or in other dataclasses, to any function of this package.

```python
import dataclasses
import torch
from d9d.core import pytree


@dataclasses.dataclass
class Batch:
    tokens: torch.Tensor
    mask: torch.Tensor
    doc_id: str  # Non-tensor fields are allowed


batch = Batch(tokens=torch.zeros(8), mask=torch.ones(8), doc_id="doc-42")

# Every tensor field moves to the GPU; other fields stay unchanged.
on_cuda = pytree.tree_map_only(torch.Tensor, lambda t: t.cuda(), batch)
```

## API Reference

::: d9d.core.pytree
