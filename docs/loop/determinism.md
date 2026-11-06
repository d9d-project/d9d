# Random Numbers in the Loop

## About

d9d derives a seed for each call of user code from `determinism.base_seed`. The loop runs the call with the global generators seeded from that seed and restores them afterwards. So the numbers of a call depend on the base seed and the position of the call, not on earlier code.

## How Seeds Are Derived

The loop runs each call of user code in a [fork of the global generators](../rng/index.md#forks-of-the-global-generators). `configure()`, `train()`, `infer()` and `export()` fork a child of `SeedScope.root(base_seed)`. Inside, each step and each call forks a child of the innermost fork. Its keys are the name of the call and its counters, such as the step, the stage and the microbatch.

The task hooks and the model pass also take the `dp` coordinate of the rank in the [base domain](../dist_context/index.md#domains) into the key. The `dp` ranks work on other samples. The model pass also takes the `cp` coordinate, because the `cp` ranks work on other parts of the same samples. It forks per stage and microbatch, so the dropout masks do not depend on the pipeline schedule. These ranks draw the same numbers:

| Call                     | Ranks that draw the same numbers                     |
|:-------------------------|:-----------------------------------------------------|
| `DataProvider`           | All ranks                                            |
| `reset_parameters()`     | All ranks of one pipeline stage                      |
| `build_forward_inputs()` | All ranks of one data-parallel shard, in every stage |
| The model pass           | Ranks that differ only in `tp`                       |

`reset_parameters()` runs with one seed per stage. So the `DTensor` RNG gives equal values on the replicas of a parameter and different values on its shards. The handlers of one event share a fork and draw in the order of subscription.

A seed set in user code, e.g. with `torch.manual_seed()`, holds only until the call returns. To make one draw independent of the draws before it, wrap it in `fork_rng_child()` with its own keys.

### Resume

A checkpoint holds no state of the global generators. The keys of a call give its seed. So a resumed job draws the same numbers as an uninterrupted one, if user code follows the rules below.

## Rules for User Code

d9d does not detect code that breaks these rules.

*   **Data iteration**: `__iter__()`, `__getitem__()` and the collator can run on the prefetch thread, in parallel with the other calls. They must not draw from the global generators. Seed a `torch.Generator` from the `seed_scope` or `shard_seed_scope` of the data provider context instead.
*   **Decisions that must match across ranks**: Draw them in `build_forward_inputs()`, e.g. an MLM mask or stochastic depth per sample. All ranks that hold a part of the sample draw the same numbers there.
*   **Parameters**: Do not run a random op on a parameter in the model pass. The seed of the pass differs across `dp` and `cp`, so the replicas would diverge.
*   **Python `random` and NumPy**: d9d does not seed them and does not set `PYTHONHASHSEED`. Seed them yourself, or create a generator from a seed scope, e.g. `numpy.random.default_rng(scope.seed)`.
*   **Code outside the calls**: d9d does not seed the global generators at start. Code before `configure()` or after `train()` seeds what it uses itself.

## Limits

*   **Identical numbers**: A rerun draws the same numbers only with the same `base_seed`, mesh layout, GPU model and software versions.
*   **Weights on another layout**: Another mesh layout can give other initial weights. To start from the same weights, export the initialized model once and load it as `model_stage_factory.source_checkpoint`.

## Usage

A dataset that draws per sample seeds a generator from its seed scope and the sample index. Then a sample gets the same numbers on every rank and in every worker:

```python
import torch
from torch.utils.data import Dataset

from d9d.loop.auto import AutoDataProvider
from d9d.rng import SeedScope


class CroppedDataset(Dataset):
    def __init__(self, texts: list[torch.Tensor], crop_length: int, seed_scope: SeedScope):
        self._texts = texts
        self._crop_length = crop_length
        self._seed_scope = seed_scope

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self._seed_scope.child(index).seed)
        tokens = self._texts[index]
        start = torch.randint(0, len(tokens) - self._crop_length + 1, (1,), generator=generator).item()
        return tokens[start : start + self._crop_length]

    def __len__(self) -> int:
        return len(self._texts)


data_provider = AutoDataProvider(
    dataset_factory=lambda context: CroppedDataset(texts, crop_length=512, seed_scope=context.seed_scope.child("crop")),
    collator=collate,
    config=config.data,
)
```

A data provider of your own seeds its loader from `shard_seed_scope`. See [Writing a Custom `DataProvider`](./interfaces/data.md#writing-a-custom-dataprovider).
