# AGENTS.md

Guidance for AI agents working in the **d9d** repository.

## Read these first

Most conventions are already documented. Always read relevant files (especially `CONTRIBUTING.md` before solving any task).

- @README.md - project purpose, philosophy, and what d9d is/isn't.
- @CONTRIBUTING.md - the canonical reference. Treat its Design Principles as hard rules when writing or changing code.
- `deps/0001-dep-process.md` and `deps/TEMPLATE.md` - when a D9D Enhancement Proposal is needed, how it moves from draft to merge, and what goes into each section. The `dep` skill runs the whole process.
- `pyproject.toml` - authoritative source for enabled `ruff` rules, `ty` config and the uv workspace. The dependencies of each package are in `packages/<name>/pyproject.toml`.
- `docs/` - user-facing documentation, mirrors the package layout in `packages/d9d/src/d9d/`.
  - `docs/index.md` - user-facing intro, the same as `README.md`.
  - `docs/toc.md` - annotated map of every subsystem and its docs page.

## Where things live

- `packages/` - the packages of the uv workspace. The published packages `d9d` and `d9d-kernels` are released together under the same version, each with its own `pyproject.toml` and its code in `src/`. The tests and benchmarks of each are a separate workspace package `packages/<name>-dev/`, also with the code in `src/`, installed into the development environment and never published.
  - `packages/d9d/src/d9d/` - library source. Mirror its layout when adding docs in `docs/`.
  - `packages/d9d-dev/` - the `d9d-dev` package with the tests of d9d (`import d9d_dev`). `-m local` (single process) and `-m distributed` (require `torchrun`).
  - `packages/d9d-kernels/` - the `d9d-kernels` package (`import d9d_kernels`) for GPU kernels that must not import `d9d`. It is empty for now: the kernels still live in `packages/d9d/src/d9d/kernel/`.
  - `packages/d9d-kernels-dev/` - the `d9d-kernels-dev` package with the tests and benchmarks of the kernels (`import d9d_kernels_dev`).
- `deps/` - enhancement proposals.
- `example/` - runnable training examples.

### Source layout (`packages/d9d/src/d9d/`)

Top-level packages of the library.

- `dist_context/` - the `DistributedContext` (the `DeviceMesh` source of truth), its domains and `DeviceMeshParameters`.
- `pytree/` - PyTree traversal and the `PyTree`/`TensorTree` aliases.
- `autograd/` - extensions of the autograd engine: `split_backward` (separate backward passes for input and weight gradients, and the gradient context that custom autograd functions check).
- `offload/` - moving GPU state to host memory and back (the base of sleep/wake).
- `loop/` - execution engine: the `Trainer`/`Inference` lifecycle, dependency injection, config schemas, and run/control/event machinery (`auto`, `component`, `config`, `control`, `event`, `run`, `sleep`), and the `determinism` module.
- `nn/` - modeling building blocks: attention, MoE (with its expert parallelism), normalization, heads and other layers, and `ModuleLateInit`.
- `parallel/` - horizontal parallelism: FSDP, HSDP and replicate placements, `runtime` (the `ParameterRuntime`, which gathers weights and reduces gradients from the pipeline program) and `grad_norm` (global gradient norm and clipping).
- `models/` - the model catalogue: the decoder backbone and heads, and one package per model with its Hugging Face mappers and its parallelization.
- `pipelining/` - pipeline parallelism: `api`, `stage`, `runtime` (the VM), `program` (`builder` and the `catalogue` of schedules) and `factory`.
- `model_state/` - checkpoints: `mapper` (graph-based transform engine) and `io` (streaming reader/writers).
- `data/` - distributed-aware dataset wrappers, bucketing, microbatch packing, the streams that hand packs to the device and the data protocols.
- `peft/` - parameter-efficient fine-tuning: `lora`, `full_tune`, `stack` (method stacking) and the `factory` of any PEFT method.
- `metric/` - distributed-aware metrics: `component`, `catalogue` (metric catalogue) and `AsyncMetricCollector` (synchronization on a side CUDA stream).
- `optim/` - the optimizer and LR scheduler protocols, `stochastic` (stochastic-rounding low-precision optimizers) and `lr_scheduler` (`piecewise` composable schedules).
- `tracker/` - experiment tracking: the trackers (`provider`, e.g. W&B, TensorBoard, MLflow, Aim) and `TrackerWriter`, which logs GPU values without waiting for the GPU.
- `kernel/` - custom kernels: `cce`, `moe`, `rms_norm`, `silu_mul`, `stochastic`.

## Working agreements for agents

- Always run `make lint` before considering a change done. It formats, fixes imports, lints, and type-checks. Type errors are not acceptable in core code.
- Add tests for any feature or fix. Match the existing tier (`local` vs `distributed`). Note: `make test` includes distributed tests that require an 8-GPU setup; run `make test-local` when GPUs are unavailable, and say so.
- Follow the Design Principles in CONTRIBUTING.md. PRs that violate them get rejected.
- PR titles must be Conventional Commits. Versioning is automated via Semantic Release — a wrong `type` produces a wrong release.
- Check "The DEP Process" in CONTRIBUTING.md before a change that adds public API. The `dep` skill writes the DEP.
- Do not add backward-compat shims for old PyTorch/hardware. The project intentionally targets modern APIs (`DTensor`, `DeviceMesh`).
- Only commit, push, or open PRs when explicitly asked.
