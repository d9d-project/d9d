# AGENTS.md

Guidance for AI agents working in the **d9d** repository.

## Read these first

Most conventions are already documented. Always read relevant files (especially `CONTRIBUTING.md` before solving any task).

- `README.md` - project purpose, philosophy, and what d9d is/isn't.
- `CONTRIBUTING.md` - the canonical reference. It covers:
  - Development setup.
  - The `Makefile` workflow.
  - Design Principles. Treat these as hard rules when writing or changing code.
  - Linting (`ruff`), type checking (`ty`), testing tiers, and docstring style.
  - The DEP process for major changes.
  - Conventional Commits format and the PR checklist.
- `deps/0001-dep-process.md` - when and how to write a D9D Enhancement Proposal.
- `pyproject.toml` - authoritative source for enabled `ruff` rules, `ty` config, and dependencies.
- `docs/` - user-facing documentation, mirrors the package layout in `d9d/`.
  - `docs/index.md` - user-facing intro.
  - `docs/toc.md` - annotated map of every subsystem and its docs page.

## Where things live

- `d9d/` - library source. Mirror its layout when adding docs in `docs/`.
- `test/d9d_test/` - tests. `-m local` (single process) and `-m distributed` (require `torchrun`).
- `deps/` - enhancement proposals.
- `example/` - runnable training examples.
- `packages/` - manually-built optional dependencies (see `compat-local-overrides` in [CONTRIBUTING.md](./CONTRIBUTING.md)).

### Source layout (`d9d/`)

Top-level packages of the library.

- `core/` - distributed primitives: `dist_context` (the `DeviceMesh` source of truth), `dist_ops`, `sharding` (PyTree sharding), `offload` (sleep/wake state offloading), `autograd`, `protocol`, `types`.
- `loop/` - execution engine: the `Trainer`/`Inference` lifecycle, dependency injection, config schemas, and run/control/event machinery (`auto`, `component`, `config`, `control`, `event`, `run`).
- `module/` - modeling building blocks: `base`, `block`, `model` (model catalogue), and `parallelism`.
- `pipelining/` - pipeline parallelism: `api`, `factory`, `infra` (the VM and schedules), and `training`.
- `model_state/` - checkpoints: `mapper` (graph-based transform engine) and `io` (streaming reader/writers).
- `dataset/` - distributed-aware dataset wrappers and bucketing.
- `peft/` - parameter-efficient fine-tuning: `lora`, `full_tune`, `all` (method stacking).
- `metric/` - distributed-aware metrics: `component` and `impl` (metric catalogue).
- `optim/` - optimizers, including `stochastic` (stochastic-rounding low-precision).
- `lr_scheduler/` - learning-rate schedules, including `piecewise` (composable schedules).
- `tracker/` - experiment tracking integrations (`provider`, e.g. WandB, Aim).
- `kernel/` - custom kernels: `cce`, `flash_attn`, `gmm`, `moe`, `normalization`, `stochastic`, `swiglu`, `general`.
- `internals/` - engine internals: `pipeline_state`, `grad_sync`, `grad_norm`, `metric_collector`, `determinism`, `profiling`, `state`.

## Working agreements for agents

- Always run `make lint` before considering a change done. It formats, fixes imports, lints, and type-checks. Type errors are not acceptable in core code.
- Add tests for any feature or fix. Match the existing tier (`local` vs `distributed`). Note: `make test` includes distributed tests that require an 8-GPU setup; run `make test-local` when GPUs are unavailable, and say so.
- Follow the Design Principles in CONTRIBUTING.md. PRs that violate them get rejected.
- PR titles must be Conventional Commits. Versioning is automated via Semantic Release — a wrong `type` produces a wrong release.
- Do not break public APIs without a DEP. Bug fixes and new models on existing APIs do not need one; breaking changes and new distributed strategies do.
- Do not add backward-compat shims for old PyTorch/hardware. The project intentionally targets modern APIs (`DTensor`, `DeviceMesh`).
- Only commit, push, or open PRs when explicitly asked.
