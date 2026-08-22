---
DEP: 0007
Title: MoE Routing Replay for HF Parity Tests
Author: Illarion Iov
Status: Draft
Type: Feature
Created: 2026-05-26
---

# DEP-0007: MoE Routing Replay for HF Parity Tests

## Abstract

This proposal adds a small test-only helper, `test/d9d_test/modules/helper/routing_replay.py`,
that lets HF-to-d9d gradient-parity tests for MoE models bypass d9d's own router during the
d9d forward and use the routing decisions captured from HF instead. With identical expert
assignments on both sides, the gradient comparator measures only d9d's compute path (dispatch,
grouped GEMM, silu_mul, combine, shared expert) and is no longer perturbed by ULP-level bf16
noise in upstream activations that flips a small fraction of tokens to different experts.

The helper consists of:

* a generic context manager `d9d_routing_replay(model_d9d, routing, batch_size)` that
  monkey-patches `MoELayer.forward` for the duration of a `with` block;
* a `RoutingCaptureFn` type for per-HF-family capture functions; one reference implementation
  for DeepSeek-V3 (`capture_hf_routing_dsv3`) is included;
* a `zero_hf_router_gradients(model_hf)` helper that zeroes HF's router-gate gradients after
  replay so the existing both-zero filter in the gradient comparator silently skips the
  router-gate keys (d9d's router gate gets no gradient under replay because it is bypassed).

The helper is test-only: nothing in the runtime training path changes.

## Motivation

HF-to-d9d gradient parity tests run an HF model and a d9d model from identical weights on the
same batch, then compare per-parameter gradients after a backward pass. For non-MoE models in
d9d this works in bf16 within a tight angle tolerance (`0.01`).

For MoE models — specifically DeepSeek-V3, but the failure mode is structural to any MoE — the
test fails: small part of parameter gradients fall outside the angle tolerance. After narrowing,
the root cause is not a divergence in d9d's compute path. d9d's grouped GEMM, silu_mul,
combine and shared expert are bit-equivalent to HF's eager reference when fed the same routing.
The divergence is:

1. ULP-scale bf16 noise in upstream computation (attention output, residual sums, RMSNorm)
   shifts the router-gate logits by a few ULPs.
2. For tokens whose top-k expert scores happen to lie close to a decision boundary, this is
   enough to flip the top-k selection. In practice 0.5 - 1.0% of tokens end up routed to
   different expert sets on the two sides.
3. Each flipped token contributes a per-token routed delta of order 3% in angle. Aggregated
   over a layer and propagated backward, this produces 2 - 7% gradient angle errors across
   most of the model — embeddings, attention, layer norms, FFNs — even where the local compute
   path is correct.

The flip rate cannot be driven to zero by tightening d9d's compute. It is a property of the
input distribution (routing-score margin) and the upstream noise floor (bf16 ULP). Adding
tolerance to the gradient comparator (e.g. raising the angle threshold to 0.1) hides real
compute regressions in the very modules we want to test.

## Design Proposal

### `RoutingCaptureFn`

```python
RoutingCaptureFn = Callable[
    [nn.Module],
    tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], list],
]
```

A capture function is HF-model-family specific because the path to the router and the way it
exposes its decisions differ between architectures. It takes the HF model, registers forward
hooks on each MoE layer's router, and returns:

* `routing` — a dict that is populated as HF's forward runs, keyed by layer index. Each
  entry is a `(indices, weights)` pair where `indices` has shape `(B*S, top_k)` and `weights`
  has the same shape, both detached and cloned.
* `handles` — the PyTorch hook handles the caller must remove after the HF forward (so the
  hooks do not stay live across other tests).

The reference DSv3 implementation hooks each `mlp.gate` module and invokes
`mlp.route_tokens_to_experts(logits)` inside the hook to obtain the same `(indices, weights)`
that HF would have used internally. It accepts both the wrapped (`DeepseekV3ForCausalLM`,
`...ForSequenceClassification` — submodules under `model.layers.*`) and the bare
(`DeepseekV3Model` — submodules under `layers.*`) HF model classes.

### `d9d_routing_replay(model_d9d, routing, batch_size)`

A context manager that monkey-patches `MoELayer.forward` for the duration of a `with` block.
Inside the block, every `MoELayer` instance in `model_d9d.model.layers` whose layer index
appears in the `routing` dict skips its own router and uses the captured `(indices, weights)`
pair instead. MoELayers not present in the dict fall through to the unpatched forward.

The patched forward:

1. preserves d9d's shared-expert path (shared expert is independent of routing);
2. reshapes the captured routing to align with the current d9d sequence length — HF is
   typically run on the full sequence `(B, S_hf)` while d9d (for causal LM tests) runs on
   the shifted sequence `(B, S_hf - 1)`. The replay trims along the sequence dim so each
   `(batch, position)` pair gets the routing originally captured for that same token;
3. otherwise dispatches, runs grouped experts, combines and reshapes identically to the
   real `MoELayer.forward`.

The monkey-patch must wrap **both forward and backward**. Activation checkpointing recomputes
the forward during backward; if the patch has been restored by then, recomputation runs d9d's
own router (different decisions ⇒ different intermediate tensor shapes than the saved ones
from the original forward), and `torch.utils.checkpoint` errors with a metadata mismatch.

### `zero_hf_router_gradients(model_hf)`

Under replay, d9d's router gate is bypassed and accumulates no gradient. HF's router gate
runs normally and does. The gradient comparator already has a both-zero filter that silently
skips parameter keys whose gradient is zero on both sides; this helper zeros HF's matching
gates after the backward so that filter triggers for the router-gate keys.

The matched pattern is `^(?:model\.)?layers\.\d+\.mlp\.gate\.weight$`, covering both wrapped
and bare HF model classes.

Router correctness is not tested by the replay path — it is validated separately by the
standalone router unit test (`test_sigmoid_grouped_router.py` for sigmoid grouped routing).

## Usage

In a parity test that targets an MoE-routing-sensitive model:

```python
import contextlib

from d9d_test.modules.helper.routing_replay import (
    d9d_routing_replay,
    zero_hf_router_gradients,
)

MOE_ROUTING_CAPTURE: dict[ModelCatalogue, RoutingCaptureFn] = {
    ModelCatalogue.DEEPSEEK_V3: capture_hf_routing_dsv3,
}

def test_consistent_to_hf(model_type, model_factory_d9d):
    model_hf = HF_MODEL_FACTORY[model_type]()

    capture_fn = MOE_ROUTING_CAPTURE.get(model_type)
    if capture_fn is not None:
        hf_routing, capture_handles = capture_fn(model_hf)
    else:
        hf_routing, capture_handles = None, []

    try:
        outputs_hf = model_hf(...)
        outputs_hf.loss.backward()
    finally:
        for h in capture_handles:
            h.remove()

    model_d9d = model_factory_d9d(stage)
    clone_module_weights(from_module=model_hf, to_module=model_d9d, map_with=...)

    replay_ctx = (
        d9d_routing_replay(model_d9d, hf_routing, batch_size)
        if hf_routing is not None
        else contextlib.nullcontext()
    )
    with replay_ctx:
        outputs_d9d = model_d9d(...)
        outputs_d9d["loss"].backward()

    if hf_routing is not None:
        zero_hf_router_gradients(model_hf)

    assert_mapped_gradients_close(...)
```

Models that are not routing-sensitive (i.e. not in `MOE_ROUTING_CAPTURE`) take the
`contextlib.nullcontext()` branch and behave exactly as before.

## Backward Compatibility

None. This adds a test-only helper module and a per-test opt-in. No production code changes.
Existing parity tests that do not use the helper are unaffected.

## Alternatives Considered

* **Tighten d9d's compute toward HF until routing flips disappear.** Investigated; the flip
  rate is bounded below by bf16 ULP noise on token activations near the routing-score margin.
  No achievable kernel-level change drives it to zero. Verified op-by-op against HF's eager
  reference: d9d's grouped GEMM and combine are bit-equivalent when fed the same routing.

* **Loosen the angle tolerance.** Rejected. The threshold that hides routing flips (~0.1 on
  some parameters) is large enough to hide real compute regressions in attention, layer norm,
  and FFN kernels, which is exactly what this test exists to catch.

* **Force fp32 on the upstream activations.** Eliminates the flips but is unrepresentative of
  training, defeats the purpose of running parity in bf16, and would require the production
  model to support a "test mode" with elevated precision.

* **Run d9d's router in fp32 from fp32-elevated logits.** Already done — DSv3's router runs
  its gate matmul in fp32. The flips originate one level upstream of the router (in the
  hidden states the router consumes), not in the router itself.

* **Apply replay at the loss level (e.g. compare only routed-output activations).** Discards
  the gradient parity signal for everything downstream of the router, which is where most
  real regressions would show up.
