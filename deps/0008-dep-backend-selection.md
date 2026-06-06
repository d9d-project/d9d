---
DEP: 0008
Title: Backend Selection Mechanism
Author: Maksim Afanasyev @mrapplexz
Status: Draft
Type: Feature
Created: 2026-06-06
---

# DEP-0008: Backend Selection Mechanism

## Abstract

Several modeling primitives in d9d can be computed by interchangeable implementations - one *operation* with
multiple *backends* (e.g. scaled dot-product attention, MoE dispatch-combine). Today these choices are hardcoded
(`self.kernel = FlashSdpa()`), with no uniform way to pick one. This DEP introduces a small, repeatable pattern: 
per backend family, a closed Pydantic discriminated union of configs, a `build_*` factory, and family-defined 
auto-detection. Modules take one `Any<Family>Backend | None` argument and delegate to the factory; resolution is 
`explicit config > D9D_BACKEND_AUTO_* env var (a JSON backend config) > auto-detection`.

## Motivation

d9d is a white-box framework: users pick, swap, and tune the kernels their models run on. The current state prevents
this. For instance, SDPA is hardcoded: `GroupedQueryAttention` and `MultiHeadLatentAttention` both do 
`self.kernel = FlashSdpa()` - no seam to substitute another kernel without editing source.

These cases will multiply. Without one agreed pattern, each new kernel reinvents its own selection surface. We need
a single mechanism that lets a backend (1) ship a programmatic default so existing models keep working, (2) be
overridden per-module via config, and (3) be forced to a global default via an environment variable - all without
global mutable state or singletons.

## Design Proposal

### Per-family layout

Each family owns a co-located package so one factory and union are shared by every module in the family (e.g.
`sdpa` is used by both `GroupedQueryAttention` and `MultiHeadLatentAttention`):

```
d9d/module/block/attention/sdpa/
    __init__.py    # re-exports AnySdpaBackend, build_sdpa_backend, configs
    config.py      # backend configs + AnySdpaBackend discriminated union
    factory.py     # build_sdpa_backend(...) + auto-detection
    flash.py       # FlashSdpa (existing implementation)
```

### Config union + factory

Each backend is a Pydantic config with a `Literal` discriminator (its stable tag). A union wraps them; a single
free `build_*` factory resolves `Any<Family>Backend | None` with an exhaustive `match`. Backend configs are pure
description (no behavior) and `model_dump`-able for hparam logging.

```python
# config.py
class FlashAttention4SdpaBackend(BaseModel):
    kind: Literal["flash_attention_4"] = "flash_attention_4"
    # backends may carry their own tuning fields here

AnySdpaBackend = Annotated[FlashAttention4SdpaBackend, Field(discriminator="kind")]

# factory.py
def build_sdpa_backend(config: AnySdpaBackend | None, *, num_sinks: int | None, window_size: int | None) -> nn.Module:
    resolved = config if config is not None else _auto_detect_sdpa_backend()
    match resolved:
        case FlashAttention4SdpaBackend():
            return FlashSdpa(num_sinks=num_sinks, window_size=window_size)
        case _:
            raise ValueError(...)
```

The factory's non-config keyword args are module-supplied *build* parameters (e.g. `num_sinks`, `window_size`) that
shape this particular layer; the backend config carries only *which backend* and that backend's own tuning fields.

### Auto-detection + global env override

Auto-detection is the family's fallback when `config is None`, and the single place the env var is read - keeping
`os.environ` at one boundary, validated through the union:

```python
_ENV_VAR = "D9D_BACKEND_AUTO_SDPA"

def _auto_detect_sdpa_backend() -> AnySdpaBackend:
    forced = os.environ.get(_ENV_VAR)
    
    if forced is not None:
        return TypeAdapter(AnySdpaBackend).validate_json(forced)
    
    # programmatic defaults:
    if some_cuda_check:
      return FlashAttention4SdpaBackend()
    
    return FlashAttention2SdpaBackend()
```

* **Env var:** `D9D_BACKEND_AUTO_<FAMILY>`; value is a **JSON backend config** validated through the union, so it
  can both select a backend *and* set that backend's tuning fields, e.g.
  `D9D_BACKEND_AUTO_SDPA='{"kind": "flash_attention_4"}'`. Malformed JSON or an unknown `kind` fails fast via
  Pydantic.
* **Precedence ("explicit wins"):** an explicit non-`None` config is never overridden by the env var. The env var
  only replaces the *auto-detection* result - a global default for everything left on auto.

Some families need context to auto-detect - such factories take extra auto-detect arguments.

### Module and model integration

Every module in a family takes one optional argument, defaulting to `None`, and delegates to the factory. Models
carry the same `Any<Family>Backend | None` field in their Pydantic params and thread it down; per-layer params give
per-layer granularity. Because everything defaults to `None`, existing models build exactly what they build today.

```python
class GroupedQueryAttention(nn.Module, ModuleLateInit):
    def __init__(self, ..., sdpa_backend: AnySdpaBackend | None = None) -> None:
        ...
        self.kernel = build_sdpa_backend(sdpa_backend, num_sinks=None, window_size=None)

class Qwen3DenseLayerParameters(BaseModel):
    ...
    sdpa_backend: AnySdpaBackend | None = None  # threaded into GroupedQueryAttention by the decoder layer
```

### Extensibility is closed by design

The union is **closed**: backends are fixed in `config.py` and resolved with an exhaustive `match (case _:
raise)`. This matches every other polymorphic config in the repo.

Users never need an out-of-tree *selectable* backend: a backend for d9d's own modules is an in-tree catalogue
contribution, and a bespoke kernel in a user's own model lives in their own module code, bypassing the union entirely.

### Experimentation seam

To iterate on a custom kernel without extending the union, swap the backend-holding attribute (`self.kernel` on
attention, `self._communicator` on `MoELayer`) - a duck-typed object satisfying the family's contract. This injects
an instance into specific modules; it does not register a selectable backend. Two timings:

* **Parameter-bearing kernels:** swap on the meta device in `ModelProvider.initialize_model_stage`, before
  `to_empty()` / `reset_parameters()`, so the kernel's params materialize and the state-dict mapper can account for
  them.
* **Parameter-free kernels:** swap via `EVENT_TRAIN_MODEL_STAGES_READY` (DEP-0003), decoupled from model/provider
  code. This fires *after* materialization, so it is unsuitable for parameter-bearing kernels.

## Usage

```python
# Default: nothing specified -> auto-detection -> FlashAttention 4, exactly as before.
Qwen3DenseForCausalLMParameters(model=Qwen3DenseParameters(...))

# Granular: pin this model's attention backend.
from d9d.module.block.attention.sdpa import FlashAttention2SdpaBackend
Qwen3DenseLayerParameters(..., sdpa_backend=FlashAttention2SdpaBackend())
```

```bash
# Global: set the "auto" default for every module left on auto (explicit configs unaffected).
# The value is a JSON backend config, so it can also set the backend's own tuning fields.
export D9D_BACKEND_AUTO_SDPA='{"kind": "flash_attention_2"}'
```

## Backward Compatibility

Fully backward compatible. Every new constructor argument and model-parameter field defaults to `None`, routing to
auto-detection that reproduces today's choices. New env vars are opt-in. No public signature is removed or retyped.

## Alternatives Considered

* **Post-construction swap pass** (walk `named_modules()` and replace backends).
  Rejected: "build wrong then swap" is less direct than building correctly.
* **Open registry / plugin entry points.** Rejected: see "Extensibility
  is closed by design".
* **Bare `Enum`/`Literal` instead of a Pydantic union.** Rejected: cannot carry per-backend parameters and bypasses
  validation of untrusted input (config, env string). The union gives both, while staying `model_dump`-able.
* **Global settings singleton (`torch.backends`-style).** Rejected: violates dependency injection principles and
  does not allow for fine-grained kernel control.
* **Selection on `DistributedContext`.** Rejected: overloads a distributed-primitives object with a modeling concern
  and doesn't deliver per-module granularity.
