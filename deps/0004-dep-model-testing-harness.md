---
DEP: 0004
Title: Model Testing Harness
Author: Maksim Afanasyev @mrapplexz & Roman Malkiv @RMalkiv
Status: Implemented
Type: Feature
Created: 2026-03-24
---

# Model Testing Harness

## Abstract

This DEP proposes a new way how full-model tests are structured.

The goal is to remove mechanical boilerplate (sequence batch construction, deterministic seeding, weight cloning,
distributed model setup) from parity tests, eliminate the copy-pasting of identical test logic across different model
architectures.

This design introduces a new approach to model testing:

- Generic reusable helper functions live under `test/d9d_test/modules/helper/`.
- Full-model tests are grouped by task (e.g., `sequence/causal_lm`, `sequence/classification`) rather than architecture.
- A centralized `ModelCatalogue` registers factories, `ModelStateMapper`s, and parallelization functions for each
  architecture.
- Tests are written *once* per task and parameterized over the catalogue.

The result is a highly maintainable test suite where adding a new model architecture requires zero new test files - only
adding a registry entry.

## Motivation

### Problem statement

Before this refactor, our test suite was architecture-centric. For every new model family, we duplicated test logic:

1. `test_lm.py` and `test_cls.py` were copy-pasted into both architecture folders.
2. The complex Pipeline schedule wiring and distributed microbatch slicing logic were identical but duplicated.
3. Standalone `util.py` files held scattered, imperative weight-cloning scripts.
4. Adding a new task required writing it N times for N architectures.

### Goals

- Shift from architecture-centric tests to task-centric ones.
- Write test logic (forward/backward, loss calculation, distributed schedule execution) exactly once per task.
- Parameterize over architectures using a static registry (`ModelCatalogue`).
- Centralize repeated plumbing (seeding, batching, assertions) in `helper/` as regular functions.
- Keep the local test behavior explicit (no "god" classes) - the test should be readable.

---

## How Model Tests Are Written Now

A Task Directory (e.g., `test_d9d/modules/model/sequence/causal_lm/`) consists of:

1. `batch.py`: Task-specific data structures.
2. `catalogue.py`: The architecture registry.
3. `test_hf.py`: Mapped forward/backward parity with HuggingFace.
4. `test_distributed.py`: Self-consistency across parallelism setups.

...and an entry to a `ModelCatalogue` enumeration:

```python
class ModelCatalogue(StrEnum):
    QWEN3_MOE = auto()
    QWEN3_DENSE = auto()
```

### Task-specific data structures

These are just simple callable functions that create all the tensors required to run a task-specific test.

```python
@dataclass
class CausalLMBatch:
    ...
    labels: torch.Tensor


def build_causal_lm_batch(device: torch.device | str = "cuda") -> CausalLMBatch:
    ...
    labels = batch.input_ids.clone()
    ...
    return CausalLMBatch(..., labels=labels)
```

### The architecture registry

When adding a new architecture, you **do not write new tests**. You simply register the model in the task's
`catalogue.py`.

The catalogue maps the `ModelCatalogue` enum to everything a generic test needs to run that architecture:

- HuggingFace Model Factory
- d9d Model Factories (e.g., with and without checkpointing)
- `ModelStateMapper` to clone weights from HF to d9d
- `ModelStateMapper` to clone weights from d9d to HF
- The `parallelize_*` function for distributed tests

```python
import transformers as tr

from d9d_test.modules.model.sequence.catalogue import ModelCatalogue

# 1. Register HF Builder
HF_MODEL_FACTORY_CAUSAL_LM = {
    ModelCatalogue.QWEN3_MOE: hf_model_factory(tr.Qwen3MoeForCausalLM, config=...),
    ModelCatalogue.QWEN3_DENSE: hf_model_factory(tr.Qwen3ForCausalLM, config=...),
}

# 2. Register d9d Builder
D9D_MODEL_FACTORIES_CAUSAL_LM = {
    ModelCatalogue.QWEN3_MOE: [d9d_model_factory(Qwen3MoEForCausalLM, ...)],
    ModelCatalogue.QWEN3_DENSE: [d9d_model_factory(Qwen3DenseForCausalLM, ...)],
}

# 3. Register Mappers
HF_TO_D9D_MAPPER_CAUSAL_LM = {
    ModelCatalogue.QWEN3_MOE: mapper_from_huggingface_qwen3_moe_for_causal_lm(...),
    ModelCatalogue.QWEN3_DENSE: mapper_from_huggingface_qwen3_dense_for_causal_lm(...),
}

# 4. Register distributed function
D9D_PARALLELIZE_FN = {
    ModelCatalogue.QWEN3_MOE: parallelize_qwen3_moe_for_causal_lm,
    ModelCatalogue.QWEN3_DENSE: parallelize_qwen3_dense_for_causal_lm,
}
```

### Mapped forward/backward parity with HuggingFace

Because of the catalogue, the actual parity test logic is written purely functionally and parameterized over the
ModelCatalogue registry using `@pytest.mark.parametrize`.

A single test function handles:

1. Generating the task batch.
2. Instantiating the HF model and d9d model.
3. Copy parameters seamlessly from HF to d9d using `clone_module_weights` and the catalogue's `ModelStateMapper`.
4. Executing task-specific forward and loss logic explicit to the task.
5. Calling assertions for loss and full model gradient verification (gradients could be unified using a
   `ModelStateMapper`).

### Self-consistency across parallelism setups

Distributed self-consistency tests follow the exact same parameterized pattern over architectures and device meshes.

The test body dynamically pulls the model factory and parallelization function from the catalogue.
It explicitly constructs the GPipe pipeline schedule, shards the data, and invokes `schedule.step()`.

## Secondary Rules for Utilities

To keep the `helper` library from bloating, the following rules apply:

1. **Test-local State Mapping for Blocks:** Standalone `util.py` files inside block test directories have been removed.
   If a block needs mapped weights (e.g. grouped-query attention, embedding splits), define the `ModelStateMapper`
   natively inside the test file. Do not define public utility functions for individual block mappings.
2. **Native Assertions:** Tests must use `torch.testing.assert_close` explicitly. Centralize structural assertions only
   if they contain complex calculations (e.g., `assert_mapped_gradients_close` handling structural name-mapping,
   norm/angle calculation).
3. **No Heavy Context Objects:** The `helper.distributed` module provides static functions (`microbatch_slice`,
   `all_reduce_over_mesh_groups`) but deliberately avoids creating a generic "PipelineTestHarness" object. The
   distributed control flow (hooks, buffers, schedule `step()`) must remain visible in the test body.

## Alternatives Considered

### Architecture-Centric Layout with Shared Utilities

**Rejected.** We considered keeping model-specific directories (e.g., `model/qwen3_dense/test_lm.py`) but moving
their inner logic into shared functions. However, this still required creating and maintaining identical test files
every time a new architecture was added. The `ModelCatalogue` handles N architectures with 1 test files,
scaling better.

### Large Generic Scenario / State Object

**Rejected.** We considered creating a generic test runner class (e.g., `DistributedTestHarness`) that inherently owns
the models, callbacks, accumulators, and schedule state. This hides orchestration logic entirely; when a distributed
test fails, it becomes very difficult for developers to trace pipeline steps or manually tweak it.
