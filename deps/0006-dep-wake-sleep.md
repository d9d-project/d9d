---
DEP: 0006
Title: Trainer Wake / Sleep APIs for Colocated RL
Author: Daniil Sergeev @DaniilSergeev17
Status: Draft
Type: Feature
Created: 2026-05-22
---

# DEP-0006: Trainer Wake / Sleep APIs for Colocated RL

## Abstract

This proposal introduces a `sleep()` / `wake()` API on the `Trainer` that releases GPU-resident training state to
host memory and restores it on demand. The goal is to support **colocated reinforcement learning**, where the same
physical GPUs alternate between two regimes — *policy update* (forward/backward/optimizer step in d9d) and *rollout*
(generation by an inference engine such as vLLM or SGLang). Each regime individually consumes most of HBM, so they
cannot coexist on a device; a clean hand-off is required.

The design adds an `Offloadable` protocol implemented by every training subsystem that holds GPU memory — tracked
modules, optimizer, gradient manager, and the distributed context's NCCL communicators. `Trainer.sleep(tags=...)` /
`wake(tags=...)` fan the call out to those subsystems. Two tags drive the selection: `TENSOR_STATES` (model,
optimizer and gradient buckets, offloaded as a unit) and `COMMS` (NCCL groups, opt-in). Four lifecycle events
(`EVENT_TRAIN_SLEEP_PRE/POST`, `EVENT_TRAIN_WAKE_PRE/POST`) let user code wire the paired rollout-engine transition
through the existing event bus. Sleep is reversible, granular, and observable — there is no global "suspended" flag
and no implicit interaction with the `train()` loop.

## Motivation

### Colocated RL is a first-class regime

The current `Trainer` assumes exclusive ownership of its GPUs for the lifetime of `train()`. For supervised training
this is correct. For RL post-training (GRPO, RLVR in general) it fails: a single GPU cannot simultaneously host d9d state
(model shards, optimizer state, gradient buckets, NCCL buffers) **and** the rollout engine (a separate
inference-layout copy of the weights, KV cache, paged-attention pools, its own communicators). The two regimes must
take turns, and the framework needs an explicit hand-off.

### Why a dedicated API and not just `.to('cpu')`

Even a complete tensor-walking `.to('cpu')` — one that reaches every parameter, buffer and optimizer slot — is
insufficient:

- **DTensor coherence.** Parameters and optimizer state are `DTensor`s whose `DeviceMesh` is bound to the `cuda`
  device type. Moving a local shard to CPU while the mesh still claims a CUDA device leaves the wrapper incoherent —
  the next collective asserts. The wrapper must be dropped for the sleep and rebuilt on wake.
- **Communicators are not tensors.** NCCL groups own per-channel buffers and flight pools that never appear in
  `parameters()` or `optimizer.state`; no `.to()` can reach them.
- **Allocator cache.** After the tensors move, the caching allocator still holds the freed blocks. An explicit
  `synchronize()` + `empty_cache()` is mandatory.
- **Hand-off.** Whether the rollout engine has finished is a user-code handshake; the framework must expose sync
  points (events), not bury them in `.to()` semantics.

A dedicated API gives d9d a single place to encode all four.

## Design Proposal

### The `Offloadable` protocol

A narrow protocol describes any subsystem that owns device memory and can be rehydrated from host memory.

```python
# d9d/core/offload.py

@typing.runtime_checkable
class Offloadable(typing.Protocol):
    def offload(self, ctx: OffloadContext) -> None: ...
    def onload(self, ctx: OnloadContext) -> None: ...
    def is_offloaded(self) -> bool: ...
```

`offload` followed by `onload` must be observationally a no-op: parameter identities (`id(param)`), optimizer state
keys, DTensor placements, gradient hooks and dtypes are all preserved; only the underlying storages are reallocated.
`OffloadContext` / `OnloadContext` are small frozen dataclasses carrying the distributed context, the target device,
and a `pin_memory` flag — defaulting to `False`, since pinning a multi-hundred-GB model plus optimizer would lock
too much host RAM.

### DTensor migration

Parameters and optimizer state can both be `DTensor`. A single pair of module-level helpers — `offload_tensor` /
`onload_tensor` — encodes the migration once, and **the DTensor wrapper instance is preserved across the
transition** — `id(dtensor)` is the same before offload and after wake, as are its `device_mesh`, `placements`,
global `shape`, and global `stride`. This matters because user code (a `BaseTask`, a metric, a frozen reference
held outside the `Offloadable` graph) routinely binds a `DTensor` to a local attribute; rebuilding the wrapper
would silently strand those references on a stale object whose next collective asserts.

To preserve identity, the helpers operate on the wrapper's local shard in place rather than constructing a new
`DTensor`. `offload_tensor` allocates a host-side mirror, copies the local shard into it, and rebinds the
wrapper's underlying local storage to that mirror (`dtensor._local_tensor.data = host_mirror`, the same `.data`
swap `TrackedModules` performs on plain tensors). `onload_tensor` reverses the swap with a fresh device buffer.
No DTensor metadata is recorded or rebuilt — `device_mesh`, `placements`, global `shape`, and global `stride`
live on the wrapper and never leave it. No CPU twin `DeviceMesh` is created — between offload and wake the
wrapper is unused, and its local shard happens to reside on host. Plain (non-distributed) tensors take the same
`.data` swap path with no DTensor-specific branch.

### Subsystems

Four components implement `Offloadable`. Each holds its host-side mirror in a private attribute and rejects double
offload / double onload with `RuntimeError`.

- **`TrackedModules`** — swaps `.data` of every parameter and buffer (persistent and non-persistent alike) for a
  host copy. Preserves `id(tensor)` for both plain tensors and `DTensor`s (see [DTensor migration](#dtensor-migration)
  for how the wrapper instance survives), so optimizer keys, gradient hooks, and any external references — e.g.
  a `BaseTask` that captured `model.lm_head.weight` at configure-time — stay valid.
- **`PipelinedOptimizer`** — moves every tensor entry of the inner optimizers' `state` to host, keyed by
  `(param, key)`. Non-tensor entries (e.g. `step`) and `param_groups` are untouched.
- **`GradientManager`** — releases the `GradientSynchronizer` bucket buffers (`unbind()`) and resets the residual
  loss accumulator on offload; re-runs the install setup (`bind()`) on onload, reproducing the install/uninstall
  transition that already wraps the train loop.
- **`DistributedContext`** (tag `COMMS`) — destroys the process groups, retaining the mesh topology so they can be
  rebuilt with identical dim names on onload.

`TENSOR_STATES` drives the first three together — in practice they are always offloaded as a unit. `COMMS` drives
the last one and is a separable axis. Components that hold no meaningful GPU state (`Stepper`, `JobLogger`,
`StateCheckpointer`, schedulers, the data loader, …) are intentionally not `Offloadable`.

### `Trainer.sleep()` / `wake()`

The trainer is the single public entry point; it fans the call out to its `Offloadable` subsystems in the right
order, surrounded by events.

```python
class SleepTag(StrEnum):
    TENSOR_STATES = "tensor_states"   # model + optimizer + gradient buckets, as a unit
    COMMS = "comms"                   # NCCL process groups; opt-in

DEFAULT_SLEEP_TAGS = frozenset({SleepTag.TENSOR_STATES})

class Trainer:
    def sleep(self, tags: Iterable[SleepTag] = DEFAULT_SLEEP_TAGS) -> None: ...
    def wake(self, tags: Iterable[SleepTag] = DEFAULT_SLEEP_TAGS) -> None: ...
    def is_sleeping(self, tag: SleepTag) -> bool: ...
```

Both calls are collective (every rank invokes with identical tags) and idempotent per tag (requesting a transition
already in effect is a no-op).

**Ordering.** On `sleep`, `TENSOR_STATES` offloads gradient manager → optimizer → tracked modules (buckets first so
none holds a stale `param.grad`; the model last so the allocator coalesces before `empty_cache`); `wake` reverses
it. After the offload pass `Trainer.sleep` calls `torch.cuda.synchronize()` then `torch.cuda.empty_cache()` — in
that order — so freed blocks return to the driver for the colocated engine.

**Sleep is illegal during an in-flight gradient accumulation** (between `EVENT_TRAIN_FORWARD_BACKWARD_PRE` and
`_POST`), because partial accumulation lives in the bucket counters. A counter in `GradientManager`, incremented by
`add_loss_with_weight` and reset by `zero_grad`, enforces this; `sleep()` raises `RuntimeError` if it is non-zero.
`EVENT_TRAIN_STEP_POST` is the canonical safe sleep point — by the time it fires, `zero_grad` has run and the
counter is zero. No change to `train()` is required.

### `COMMS` is a second phase

The `COMMS` tag is fully *specified* here so the API surface is stable, but its *implementation* is deferred to a
follow-up PR. `TENSOR_STATES` already recovers the overwhelming majority of HBM and has no
fragile dependencies, whereas NCCL teardown has known upstream caveats — rapid destroy/init races, incomplete
memory release, hangs when NCCL ops were captured into a CUDA graph — that need real hardening. Until that phase
lands, `sleep(tags={SleepTag.COMMS})` raises `NotImplementedError`, and `COMMS` is excluded from
`DEFAULT_SLEEP_TAGS`. Note that once `COMMS` is offloaded no NCCL barrier exists, so no d9d collective is valid in
the rollout window — the user code coordinates there.

### Lifecycle events

Four events join the catalogue, following the existing `*_PRE` / `*_POST` convention and carrying an
`EventSleepContext` (the `frozenset[SleepTag]` involved in the transition):

- `EVENT_TRAIN_SLEEP_PRE` — before any offload; state still resident on GPU.
- `EVENT_TRAIN_SLEEP_POST` — after offload and `empty_cache`; the GPU is free. Subscribe here to wake the rollout
  engine.
- `EVENT_TRAIN_WAKE_PRE` — before any restore. Subscribe here to put the rollout engine to sleep.
- `EVENT_TRAIN_WAKE_POST` — after every subsystem is back on GPU.

The framework deliberately ships **no** `RolloutEngine` abstraction — engines (vLLM, SGLang, custom HTTP servers)
differ too widely in lifecycle, and owning one would violate the white-box-modeling principle. The user wires their
engine through these events.

## Usage

### Basic round-trip

```python
from d9d.loop.run import SleepTag, TrainingConfigurator

trainer = TrainingConfigurator(...).configure()

trainer.sleep()                                     # free GPU memory (TENSOR_STATES)
assert trainer.is_sleeping(SleepTag.TENSOR_STATES)
# ... colocated rollout engine runs ...
trainer.wake()                                      # restore
```

### Colocated RL inside the existing `train()` loop

No modification to `train()` is required — subscribe a handler to `EVENT_TRAIN_STEP_POST`, which fires after
`zero_grad` when sleep is legal:

```python
from d9d.loop.event.catalogue.train import EVENT_TRAIN_STEP_POST, EventStepContext

def install_rl_handler(trainer, rollout_engine, every_n_steps: int):
    def on_step_post(ctx: EventStepContext):
        if ctx.stepper.current_step % every_n_steps != 0:
            return
        trainer.sleep()
        rollout_engine.wake_up()
        rollout_engine.generate(...)
        rollout_engine.sleep()
        trainer.wake()

    trainer._state.event_bus.subscribe(EVENT_TRAIN_STEP_POST, on_step_post)
```

### Custom `Offloadable` subsystems

User state attached to a `BaseTask` (e.g. a frozen reference model for KL regularization) can implement
`Offloadable` and subscribe itself to the lifecycle events without framework changes:

```python
class GRPOTask(TrainTask):
    def register_events(self, ctx):
        ctx.event_bus.subscribe(EVENT_TRAIN_SLEEP_PRE, self._ref_offload)
        ctx.event_bus.subscribe(EVENT_TRAIN_WAKE_POST, self._ref_onload)
```

## Backward Compatibility

Fully backward compatible.

- `Trainer.sleep` / `wake` / `is_sleeping` are new methods; `train()` and `export()` never call them.
- `Offloadable` is a new protocol. The four existing components gain `offload` / `onload` / `is_offloaded`, but no
  existing public signature changes.
- The four new events are additive; the default no-op `register_events()` from DEP-0003 means existing subscribers
  see no new traffic.
- `TrainJobState` gains no new fields.

Supervised training scripts that never call `sleep()` behave exactly as today — the offload paths are reached only
on explicit call, and the `GradientManager` in-flight counter is a single integer folded into existing paths.

## Alternatives Considered

### `torch_memory_saver` virtual-memory pause

Pausing CUDA virtual memory keeps pointers valid and is faster than a host copy. Rejected as the primary backend: it
needs a custom CUDA shim (version/platform risk), DTensor sharding produces allocations it cannot reason about
without invasive tagging, and it hides device residency from `torch.cuda.memory_allocated()`, confusing
`JobProfiler`. The `Offloadable` protocol prescribes only a contract, not a mechanism — a memory-saver backend can
replace the host-copy implementation later behind the same API.

### Reuse the checkpointer (save → free → reload)

The checkpointer already serializes everything, but it writes through the distributed checkpoint API to disk and
rebuilds runtime objects from scratch on reload — losing parameter identity and the `GradientManager` /
`GradientClipper` bindings established at configuration time. For a once-per-step transition the disk latency
(seconds vs. tens of ms for a host copy) is unacceptable.

### A monolithic tag-less `suspend()` / `resume()`

The opposite extreme. Rejected because `COMMS` genuinely needs to be separable — its NCCL-teardown cost and
upstream fragility are unlike the plain host-copy of `TENSOR_STATES`, and many deployments (rollout engine on
different ranks or a different node) never want to pay it. Two tags is the minimal honest API; an earlier draft
split `TENSOR_STATES` further into `MODEL` / `OPTIMIZER` / `RESIDUAL`, but no real recipe offloads one without the
others, so that split was dropped.

### CPU twin `DeviceMesh` for offloaded DTensors

Rejected: building a CPU-typed `DeviceMesh` needs a separate gloo process group and a parallel set of collectives
nothing in d9d uses, and it would require recreating the `DTensor` wrapper around the CPU shard — the very
identity-breaking step the in-place `.data` swap is designed to avoid. Between offload and wake the wrapper is
unused, so the local shard simply lives on host while `device_mesh` / `placements` remain bound to the original
CUDA mesh, ready for the wake swap.
