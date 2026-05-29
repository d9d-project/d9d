---
DEP: 0006
Title: Sigmoid Grouped MoE Router
Author: Illarion Iov
Status: Implemented
Type: Feature
Created: 2026-05-18
---

# DEP-0006: Sigmoid Grouped MoE Router

## Abstract

This proposal adds `SigmoidGroupedTopKRouter` to `d9d/module/block/moe/router.py` — a router that
combines sigmoid-activated expert scores with a two-level hierarchical selection procedure
(group selection → per-group top-k). It also makes `MoELayer` router-agnostic by accepting any
router instance via constructor injection instead of constructing `TopKRouter` internally.

The new router is the routing strategy used by DeepSeek-V3, GLM-4.6, Kimi K2 / K2.6,
MiniMax-M2, and Mistral Large 3 / Small 4 — all of which are planned for d9d integration.

## Motivation

The existing `TopKRouter` uses softmax gating: token scores across all experts are normalised
jointly before top-k selection. Softmax introduces a coupling between experts that is absent in
sigmoid gating, and it does not allow per-token load-balancing corrections to be applied without
re-normalising the whole distribution.

DeepSeek-V3 and its derivative models address this with a three-part change:

1. **Sigmoid instead of softmax.** Each expert is scored independently; there is no
   inter-expert normalisation before selection.
2. **Hierarchical group selection.** Experts are divided into `n_group` equal groups. A group
   score (sum of the top-2 expert scores within the group) selects `topk_group` candidate groups
   first; the final top-k experts are drawn only from those groups. This constrains which experts
   a token can reach and simplifies load balancing.
3. **Correction bias.** An `e_score_correction_bias` buffer (one scalar per expert, not updated
   by gradient) shifts expert scores before group and expert selection. It is updated externally
   by a load-balancing controller. The final expert weights fed into the FFN use the unbiased
   sigmoid scores — the bias is selection-only.

Without this router, none of the DeepSeek-V3 family checkpoints can be loaded correctly into d9d.

## Design Proposal

### New class: `SigmoidGroupedTopKRouter`

```python
class SigmoidGroupedTopKRouter(nn.Module, ModuleLateInit):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        norm_topk_prob: bool,
    ) -> None: ...

    def forward(self, hidden_states: torch.Tensor) -> RoutingResult: ...
    def reset_parameters(self) -> None: ...
```

**Forward pass** (tokens shape `(T, dim)`):

```
scores      = sigmoid(gate(hidden_states))                      # (T, E),  float32
scores_sel  = scores + e_score_correction_bias                  # (T, E)   — for selection only
group_score = scores_sel.view(T, G, E/G).topk(2, -1)[0].sum(-1) # (T, G)
group_idx   = topk(group_score, topk_group)[1]                  # (T, topk_group)
expert_mask = scatter group_idx → (T, E) boolean mask
masked      = scores_sel.masked_fill(~expert_mask, -inf)
indices     = topk(masked, top_k)[1]                            # (T, top_k)
weights     = scores.gather(-1, indices)                        # unbiased weights
if norm_topk_prob: weights /= weights.sum(-1, keepdim=True) + ε
weights    *= routed_scaling_factor
return RoutingResult(indices, weights)
```

The `e_score_correction_bias` is a `nn.Buffer(persistent=True)`, initialised to zero. It is not
a gradient parameter; callers update it via an external load-balancing controller.

### Change to `MoELayer`

`MoELayer` is refactored to be fully router-agnostic. `router` becomes a required positional
argument; `router_renormalize_probabilities` and `top_k` are removed — both were
`TopKRouter`-specific concerns that had leaked into the wrong class.

```python
def __init__(
    self,
    hidden_dim: int,
    intermediate_dim_grouped: int,
    num_grouped_experts: int,
    router: nn.Module,
    shared_expert: SharedExpertParameters | None = None,
):
    self.router = router  # plain assignment, no branching
```

All existing call sites are updated to construct their router explicitly before passing it.
`reset_parameters` calls `self.router.reset_parameters()` and requires no changes.

### Exports

`SigmoidGroupedTopKRouter` is added to `d9d/module/block/moe/__init__.py` `__all__`.

## Usage

```python
from d9d.module.block.moe import MoELayer, SigmoidGroupedTopKRouter, TopKRouter
from d9d.module.block.moe.shared_expert import SharedExpertParameters

# DeepSeek-V3 style
moe = MoELayer(
    hidden_dim=7168,
    num_grouped_experts=256,
    intermediate_dim_grouped=2048,
    router=SigmoidGroupedTopKRouter(
        dim=7168,
        num_experts=256,
        top_k=8,
        n_group=8,
        topk_group=4,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
    ),
    shared_expert=SharedExpertParameters(intermediate_size=2048),
)

# Qwen3 style (unchanged semantics, explicit construction)
moe = MoELayer(
    hidden_dim=4096,
    num_grouped_experts=64,
    intermediate_dim_grouped=1024,
    router=TopKRouter(dim=4096, num_experts=64, top_k=4, renormalize_probabilities=True),
)
```

## Backward Compatibility

**Breaking change** to `MoELayer`: `router_renormalize_probabilities` and `top_k` are removed;
`router` is now required. All in-tree call sites are updated in the same commit. There are no
external callers in this codebase.

## Alternatives Considered

**Subclassing `TopKRouter`**: the two routers share only the `gate` linear and
`RoutingResult` return type. A base class would add overhead without benefit.

**`routing_type: Literal["softmax", "sigmoid_grouped"]` enum on `MoELayer`**:
pushes router hyperparameters into `MoELayer`, coupling it to every future routing variant.

**Optional `router=None` with internal fallback**: keeps the dead
`router_renormalize_probabilities` / `top_k` params alive and adds a conditional. The required
argument is cleaner and forces callers to be explicit about what router they are using.
