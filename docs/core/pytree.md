# PyTree Traversal

## About

The `d9d.core.pytree` package provides the framework's utilities for recursively traversing nested tensor structures ("pytrees"). It is a thin, dataclass-aware wrapper around [`optree`](https://github.com/metaopt/optree) and is used wherever the engine needs to apply an operation to every tensor in a nested structure - moving a microbatch to the device, detaching cached side-data, moving metric results to the CPU, or flattening a metric tree for logging.

It operates over the container types described by [`PyTree`](./types.md) - `dict`, `list`, `tuple` - nested arbitrarily deep, **and additionally over any dataclass**.

## Dataclasses Work Transparently

A caller can pass a dataclass - arbitrarily nested inside containers or other dataclasses - to any function in this package, and its fields are traversed as tree children.

```python
import dataclasses
import torch
from d9d.core import pytree


@dataclasses.dataclass
class Batch:
    tokens: torch.Tensor
    mask: torch.Tensor
    doc_id: str  # non-tensor bookkeeping is fine


batch = Batch(tokens=torch.zeros(8), mask=torch.ones(8), doc_id="doc-42")

# Every tensor field is moved; non-tensor fields ride along untouched.
on_cuda = pytree.tree_map_only(torch.Tensor, lambda t: t.cuda(), batch)
```

Nesting composes in every direction - a dataclass inside a dict, a list of dataclasses, or a dataclass whose fields are dicts of tensors are all traversed correctly.

## Determinism

Traversal order is deterministic:

* **`dict` keys** are traversed in **sorted** order, regardless of insertion order.
* **dataclass fields** are traversed in **declaration** order.

## API Reference

::: d9d.core.pytree
    options:
        heading_level: 4
