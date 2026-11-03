# Random Number Generation

## About

The `d9d.rng` package derives seeds from a base seed and keys, and seeds the global generators from them. A `SeedScope` is a node in a tree of seeds. The root seed is a hash of the base seed. A child seed is a hash of the parent seed and the keys of the child. So the same keys give the same seed in every process, on every rank and in every data loader worker.

## How Seeds Are Derived

Keys are strings and integers. Keys of another type or another split give another seed, e.g. `"1"` and `1`, or `("ab", "c")` and `("a", "bc")`. The hash does not depend on `PYTHONHASHSEED`, so the seeds do not change between processes, Python versions and releases of d9d.

A seed has 64 bits, the full range of `torch.Generator.manual_seed()`.

## Forks of the Global Generators

The **global generators** are the torch CPU generator and the CUDA generator of the current device. Code that passes no generator draws from them, e.g. `torch.rand()` and dropout.

`fork_rng(scope)` seeds the global generators from a scope while its context is open. On exit, it restores the states they had before, also on an exception. Forks nest. `fork_rng_child(*keys)` forks the child of the innermost open fork, so you need no scope at hand. Outside a fork, it raises an error.

The d9d loop runs every call of user code in a fork (see [Random Numbers in the Loop](../loop/determinism.md)). Inside such a call, `fork_rng_child()` makes a draw depend only on its keys, not on the draws before it.

Forks have these limits:

*   **One thread**: The global generators belong to the process. A fork on another thread, e.g. in a dataset on the prefetch thread, races with the forks of the main thread. Seed a `torch.Generator` from a scope there instead.
*   **No CUDA graph capture**: PyTorch forbids setting the generator state during capture.
*   **torch only**: A fork does not seed Python `random` or NumPy.

## Usage

Seed a generator from a scope and the keys that identify the draw:

```python
import torch

from d9d.rng import SeedScope

scope = SeedScope.root(42).child("augment")
generator = torch.Generator().manual_seed(scope.child(sample_index).seed)
noise = torch.randn(16, generator=generator)
```

In a d9d job, you get the scopes from the loop, e.g. `seed_scope` of the data provider context. See [Random Numbers in the Loop](../loop/determinism.md).

Inside a call of the loop, fork a child to give each item its own numbers, whatever was drawn before it:

```python
from d9d.rng import fork_rng_child

for prompt_idx, prompt in enumerate(prompts):
    with fork_rng_child("rollout", prompt_idx):
        completions.append(sample(model, prompt))
```

Outside the loop, open the fork yourself:

```python
from d9d.rng import SeedScope, fork_rng

with fork_rng(SeedScope.root(42).child("eval")):
    run_evaluation(model)
```

## API Reference

::: d9d.rng
