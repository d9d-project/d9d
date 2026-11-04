---
DEP: 0011
Title: Scoped Random Number Generation
Author: Maksim Afanasev @mrapplexz
Type: Feature
Implemented: #157
---

# Scoped Random Number Generation

## Abstract

d9d seeds the global random number generators once per process, and the model, the task and the providers all draw from them. So their numbers depend on unrelated code, differ between pipeline stages where they must match and repeat after a resume.

This DEP derives a seed for each call of user code from the base seed and the position of the call in the job. d9d runs the call with the global generators forked from that seed and never changes their seed otherwise. Components with their own generators derive their seeds the same way.

## Motivation

The **global generators** are the torch generators that code uses when it passes none: the CPU generator and the CUDA generator of the current device.

All defects below have one cause: the loop cannot give a call or a component its own seed.

*   **Stages read different samples**: With `shuffle: true`, `AutoDataProvider` shuffles with a sampler seeded from the global CPU generator, which `set_seeds()` seeds per stage. So the last stage computes the loss with the labels of another sample.
*   **The rounding noise ignores `base_seed`**: `AutoOptimizerProvider` passes no `seed` to `StochasticAdamW`, so all stages and all runs round with the seed `0`.
*   **The numbers depend on unrelated code**: `reset_parameters()` gives other weights if the provider code draws before it. A draw in an event handler or a rollout engine shifts all later dropout masks.
*   **A resume repeats the start**: The checkpoint holds no generator state, so after a resume dropout repeats the masks of the first steps.
*   **The data loader races with the task**: A `DataLoader` without a `generator` draws from the global CPU generator in `iter()`, which runs on the prefetch thread in parallel with the task hooks.

The first two defects are not fixed in separate PRs, because each fix needs a seed in a provider context. That seed is the abstraction this DEP introduces.

## Design Proposal

### Principles

1.  **One seed per call**: Each call of user code and each component with its own generator gets its own seed.
2.  **Explicit sharing**: The keys of a seed decide which ranks share it.
3.  **Fork, never reseed**: d9d changes the global generators only inside forks, and each fork restores them on exit.

PyTorch forces d9d to use the global generators. Dropout takes no generator, and activation checkpointing restores only the global generators during the recompute. Python `random` and NumPy have no such limit, so d9d does not seed them.

### `d9d.rng`

This DEP introduces `SeedScope` in the new package `d9d.rng`, which depends on nothing else in d9d. So datasets and `d9d.data` can take it without depending on the loop. It is a node in a tree of seeds. The root holds a hash of `base_seed`, and a child holds a hash of the seed of its parent and its keys. So the same keys give the same seed in every process.

```python
class SeedScope:
    """A node in the tree of seeds derived from the base seed of a job."""

    @property
    def seed(self) -> int:
        """The seed of this scope, in ``[0, 2**64)``. APIs that take a 32-bit seed must use it modulo ``2**32``."""

    @staticmethod
    def root(base_seed: int) -> "SeedScope":
        """Returns the root seed scope of a job, with a seed hashed from the base seed."""

    def child(self, *keys: str | int) -> "SeedScope":
        """Returns the seed scope below this one, with a seed hashed from this seed and the keys."""
```

This DEP also introduces `fork_rng()` and `fork_rng_child()` in `d9d.rng`. `fork_rng()` replaces `set_seeds()`. The loop uses both. User code can use `fork_rng_child()` to make one draw independent of the draws before it.

```python
@contextmanager
def fork_rng(scope: SeedScope) -> Iterator[None]:
    """Seeds the global generators from a seed scope while the context is open.

    On exit, the global generators return to the states they had before the context.
    """


def fork_rng_child(*keys: str | int) -> AbstractContextManager[None]:
    """Forks the global generators from a child of the seed scope of the innermost open ``fork_rng()``.

    Raises:
        RuntimeError: If no ``fork_rng()`` is open on this thread.
    """
```

### Forks of the loop

Forks nest: `fork_rng()` remembers its scope until it exits. The loop opens a fork of the root `SeedScope.root(base_seed)` for each phase: `configure()`, `train()`, `infer()` and `export()`. Inside it, each step and each call of user code forks the child `(name, *counters)` of the innermost open fork. The counters are the step, the stage and the microbatch, if the call repeats over them. Otherwise every repetition draws the same numbers. `stage` is the global index of the pipeline stage.

A call whose work differs across ranks adds the coordinate of the rank to its key. `dp` and `cp` are the rank coordinates in the batch domain, and `0` without a mesh. So every draw has the full path from the root, e.g. `("train",)`, `("step", step)`, `("dp", dp)`, `("compute_loss", microbatch)`, and two phases never share numbers.

For example:

*   `data_provider()` runs in `("data_provider",)` of `configure()`, because providers must build the same objects on all ranks.
*   `reset_parameters()` runs in `("reset_parameters", stage)` of `configure()` (see Initialization).
*   `build_forward_inputs()` runs in `("build_forward_inputs", microbatch)` of `("dp", dp)` of the step. All ranks that hold a part of a sample must make the same decision about it, such as a mask.
*   Each forward and backward pass of a stage runs in `("cp", cp, kind, stage, microbatch)` of `("dp", dp)` of the step. The `compute_context()` of the `PipelineStepHandler` opens this fork. Dropout needs other masks for other samples, parts of a sequence and layers, but the same masks across `tp`. The masks do not depend on the pipeline schedule.
*   An event runs in `("event", event.id)` of the call that triggers it, so custom events run in a fork too. The handlers draw in the order of subscription. An event of the loop draws the same numbers on all ranks. A handler that needs other numbers per rank, e.g. for rollouts, adds its rank to the keys with `fork_rng_child()`.

A random decision that must match across ranks, such as stochastic depth per sample, belongs in `build_forward_inputs()`. A random op on a parameter must not run in the model pass, because the seed of the pass differs across `dp` and `cp`. d9d does not detect code that breaks these rules.

### Initialization

With one seed per stage, the `DTensor` RNG gives equal values on replicas and disjoint values on shards. The same `base_seed`, layout, GPU model and software give the same weights. To get the same weights on another layout, initialize once and load the result as `source_checkpoint`.

`DistributedContext` creates the `DTensor` RNG tracker with `run_state_sync=False` when it builds the device meshes. Otherwise the first `DTensor` random op broadcasts the generator state of rank 0 and overwrites the seeds of the other stages. This is a stopgap, because PyTorch has no public API to create the tracker without the broadcast.

### Seed scopes in contexts

A context gets a seed scope only for code that a fork cannot cover.

The iteration over data, i.e. `__iter__()`, `__getitem__()` and the collator, can run on the prefetch thread, in parallel with the forks of the main thread. So it must not draw from the global generators and must take its seeds from the context. `InitializeDataProviderContext` gets two fields:

*   **`seed_scope`**: The child `("data",)` of the root, equal on all ranks for code before sharding.
*   **`shard_seed_scope`**: The child `("dp", dp)` of `seed_scope`, equal on the ranks of one data-parallel shard.

`AutoDataProvider` passes its context to the dataset factory.

`StochasticAdamW` takes an optional `generator`, as torch random ops do. Without it, each step draws from the global generators. In the loop, the optimizer steps in a fork, so `AutoOptimizerProvider` passes no generator and the optimizer context gets no seed scope.

### Checkpoint

The checkpoint gets no new state. The seeds of the calls follow from their keys, and the loader saves its own generator.

## Usage

A data provider derives its seeds from the context:

```python
def __call__(self, context: InitializeDataProviderContext) -> MicrobatchPackStream:
    dataset = BufferSortedDataset(..., seed=context.seed_scope.child("presort").seed)
    dataset = shard_dataset_data_parallel(dataset, context.dist_context)

    generator = torch.Generator().manual_seed(context.shard_seed_scope.child("loader").seed)
    loader = StatefulDataLoader(dataset, shuffle=True, generator=generator)
    ...
```

Code inside a call forks a child to give each item its own numbers, whatever was drawn before it:

```python
for prompt_idx, prompt in enumerate(prompts):
    with fork_rng_child("rollout", prompt_idx):
        completions.append(sample(model, prompt))
```

The model code does not change.

## Backward Compatibility

*   **Contexts**: Code that builds `InitializeDataProviderContext` itself must pass the new fields.
*   **`StochasticAdamW`**: An optional `generator` replaces the `seed` argument.
*   **`EventBus`**: `trigger()` raises outside `fork_rng()`, like every `fork_rng_child()`. The loop always opens a fork. Code that uses an `EventBus` outside the loop opens one itself.
*   **`DistributedContext`**: It replaces the `DTensor` RNG tracker of the process.
*   **`AutoDataProvider`**: The dataset factory gets `InitializeDataProviderContext` instead of `DistributedContext`.
*   **`set_seeds()`**: It no longer exists. d9d no longer seeds torch, Python `random` and NumPy at start and no longer sets `PYTHONHASHSEED`. Code outside the calls of d9d seeds what it uses itself.
*   **Seeds in user code**: `torch.manual_seed()` in a provider, a hook or an event handler holds only until the call returns.
*   **Random numbers**: The same `base_seed` gives other weights, data order and dropout masks.

## Alternatives Considered

### Seeding the global generators once per process

d9d, Megatron, torchtitan and the Hugging Face Trainer seed the global generators once per process, and torchtitan adds distinct seeds per mesh dimension. The numbers then depend on everything that drew before, so a resume cannot reproduce them without saved state. One seed per process also cannot serve per-sample decisions and dropout, because they need different rules for `cp`.

### A tracker of named CUDA states

Megatron keeps named generator states in a tracker, and each layer forks the state of its parallel region. A d9d model never reads the mesh, so it cannot choose the state.

### The `DTensor` RNG alone

torchtitan relies on the `DTensor` RNG, which derives shard offsets from the placements. It covers only ops on `DTensor` objects, but dropout and most MoE code run on local tensors.

### Generator states in the checkpoint

Megatron, OLMo-core and Accelerate save the global generators per rank. The states belong to one layout. Megatron drops them when TP, PP or the world size changes, and OLMo-core when the world size changes. Seeds per call give an exact resume without saved state.

### Generators in the hook contexts

Task hooks could take a generator from their context instead of a fork. But dropout and libraries that draw from the global generators cannot take one.

## Deferred Work

*   **Tensor parallelism**: Sharded regions need masks that differ across `tp`. This belongs to the tensor parallelism DEP (#108).
*   **Context parallelism**: The seeds of the task hooks assume that every `cp` rank sees the whole sample. A split of the sample before the hooks belongs to the context parallelism design (#5).
*   **Global offsets in stochastic rounding**: The rounding noise of `StochasticAdamW` depends on the local element index, so the shards of one parameter round with the same noise. The kernel can take the global offset of the shard from its `DTensor` placements instead.
*   **Per-sample randomness**: `AutoDataProvider` can run each sample and each collation in a fork keyed by the shard seed and the indices. This fork also seeds Python `random` and NumPy. It needs `num_workers` of at least 1, because a fork on the prefetch thread races with the forks of the main thread.
*   **CUDA graphs**: A fork cannot run inside graph capture, because PyTorch forbids setting the generator state during capture. d9d does not capture graphs yet.
