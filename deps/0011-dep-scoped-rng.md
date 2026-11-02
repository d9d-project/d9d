---
DEP: 0011
Title: Scoped Random Number Generation
Author: Maksim Afanasev @mrapplexz
Type: Feature
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

### `SeedScope`

This DEP introduces `SeedScope` in `d9d.loop.determinism`. It is a node in a tree of seeds. The root holds a hash of `base_seed`, and a child holds a hash of the seed of its parent and its keys. So the same keys give the same seed in every process.

```python
@dataclasses.dataclass(frozen=True)
class SeedScope:
    """A node in the tree of seeds derived from the base seed of a job.

    Attributes:
        seed: The seed of this scope, in ``[0, 2**64)``. APIs that take a 32-bit seed must use it modulo ``2**32``.
    """

    seed: int

    @classmethod
    def root(cls, base_seed: int) -> Self:
        """Returns the root seed scope of a job, with a seed hashed from the base seed."""

    def child(self, *keys: str | SupportsIndex) -> Self:
        """Returns the seed scope below this one, with a seed hashed from this seed and the keys."""
```

`d9d.data` and `d9d.optim` take plain `int` seeds since #138 and #139, so they do not depend on the loop.

This DEP also introduces `fork_rng()` in `d9d.loop.determinism`. It replaces `set_seeds()`.

```python
@contextmanager
def fork_rng(scope: SeedScope) -> Iterator[None]:
    """Seeds the global generators from a seed scope while the context is open.

    On exit, the global generators return to the states they had before the context.

    Args:
        scope: The seed scope to take the seed from.
    """
```

### Seed scopes of the job

The loop runs each call of user code in `fork_rng(root.child(name, *coordinates))`, where `name` is the name of the call. The coordinates are:

*   **Counters**: The call takes each counter it repeats over: `stage`, `step` and `microbatch`. Otherwise every repetition draws the same numbers. `stage` is the global index of the pipeline stage.
*   **Rank coordinates**: The call takes `pp`, `dp` or `cp` only if its numbers must differ along it. They are the rank coordinates in the batch domain, and `0` without a mesh.

For example:

*   `data_provider()` runs in `("data_provider",)`, because providers must build the same objects on all ranks.
*   `reset_parameters()` runs in `("reset_parameters", stage)`, because the `DTensor` RNG keeps replicas equal only on equal generator states.
*   `build_forward_inputs()` runs in `("build_forward_inputs", step, microbatch, dp)`. All ranks that hold a part of a sample must make the same decision about it, such as a mask.
*   `schedule.step()` runs in `("pass", step, pp, dp, cp)`. Dropout needs other masks for other samples, parts of a sequence and layers, but the same masks across `tp`.

`configure()`, `train()`, `infer()` and each step also run in forks, which cover the code between the calls, such as the saving of checkpoints. Handlers of one event share one fork and draw in the order of subscription.

A random decision that must match across ranks, such as stochastic depth per sample, belongs in `build_forward_inputs()`. A random op on a parameter must not run in the model pass, because the seed of the pass differs across `dp` and `cp`. d9d does not detect code that breaks these rules.

### Initialization

With one seed per stage, the `DTensor` RNG gives equal values on replicas and disjoint values on shards. The same `base_seed`, layout, GPU model and software give the same weights. To get the same weights on another layout, initialize once and load the result as `source_checkpoint`.

The loop creates the `DTensor` RNG tracker with `run_state_sync=False` in the fork of `configure()`. Otherwise the first `DTensor` random op broadcasts the generator state of rank 0 and overwrites the seeds of the other stages. This is a stopgap, because PyTorch has no public API to create the tracker without the broadcast.

### Data and optimizer

The iteration over data, i.e. `__iter__()`, `__getitem__()` and the collator, can run on the prefetch thread, in parallel with the forks of the main thread. So it must not draw from the global generators and must take its seeds from the context. `InitializeDataProviderContext` gets two fields:

*   **`seed_scope`**: The seed scope `("data",)`, equal on all ranks for code before sharding.
*   **`shard_seed_scope`**: The seed scope `("data", "shard", dp)`, equal on the ranks of one data-parallel shard.

A data loader must take a generator seeded from `shard_seed_scope`. A dataset that draws per sample creates a generator from its seed and the sample index. `AutoDataProvider` passes its context to the dataset factory.

`InitializeOptimizerStageContext` gets the field `seed_scope` with the seed scope `("optimizer", stage)`. `AutoOptimizerProvider` passes its seed to `StochasticAdamW`.

### Checkpoint

The checkpoint gets no new state. The seeds of the calls follow from their keys, and the loader and the optimizer save their own generators. The loop saves the loader per data-parallel rank and the optimizer per stage. The rules above make each generator equal on the ranks that share its key.

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

A task draws per-sample decisions in `build_forward_inputs()`, so all stages get the same mask:

```python
def build_forward_inputs(self, ctx: BuildForwardInputsContext) -> BuildForwardInputsResult:
    mask = torch.rand(ctx.batch.input_ids.shape, device=ctx.batch.input_ids.device) < 0.15
    ...
```

The model code does not change.

## Backward Compatibility

*   **Contexts**: Code that builds `InitializeDataProviderContext` or `InitializeOptimizerStageContext` itself must pass the new fields.
*   **`AutoDataProvider`**: The dataset factory gets `InitializeDataProviderContext` instead of `DistributedContext`.
*   **`set_seeds()`**: It no longer exists. d9d no longer seeds torch, Python `random` and NumPy at start and no longer sets `PYTHONHASHSEED`. Code outside the calls of d9d seeds what it uses itself.
*   **Seeds in user code**: `torch.manual_seed()` in a provider, a hook or an event handler holds only until the call returns.
*   **Random numbers**: The same `base_seed` gives other weights, data order and dropout masks. A job resumed from an old checkpoint keeps the saved loader and optimizer generators.

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
*   **Stateless stochastic rounding**: Derive the noise from the seed, the step and the element index. The optimizer then needs no generator state. It is not needed now, because the generator in `state_dict()` already gives an exact resume.
*   **Per-sample randomness**: A dataset wrapper that passes `__getitem__()` a generator seeded by the sample index.
*   **CUDA graphs**: A fork cannot run inside graph capture, because PyTorch forbids setting the generator state during capture. d9d does not capture graphs yet.
