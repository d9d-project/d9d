import contextlib
import re
from collections.abc import Callable, Iterator

import torch
from torch import nn

# Capture functions vary by HF MoE class. Signature:
#   (model_hf) -> (routing_dict, handles)
# where routing_dict[layer_idx] = (indices_tensor, weights_tensor) once HF's
# forward has been run, and handles is the list to remove afterward.
RoutingCaptureFn = Callable[[nn.Module], tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], list]]


def capture_hf_routing_dsv3(model_hf: nn.Module) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], list]:
    """HF DSv3 capture: hook each `mlp.gate` and run `route_tokens_to_experts`
    on the gate logits to get (topk_indices, topk_weights) per layer.

    Works for both wrapped (``DeepseekV3ForCausalLM``, ``...ForSequenceClassification``
    — layers under ``model.layers``) and bare (``DeepseekV3Model`` used by the
    embedding tests — layers under ``layers``) HF model classes.
    """
    routing: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles: list = []
    num_layers = model_hf.config.num_hidden_layers
    for i in range(num_layers):
        mlp = None
        for prefix in ("model.", ""):
            try:
                mlp = model_hf.get_submodule(f"{prefix}layers.{i}.mlp")
                break
            except AttributeError:
                continue
        if mlp is None:
            continue
        if not hasattr(mlp, "gate") or not hasattr(mlp, "route_tokens_to_experts"):
            continue

        def _make_hook(idx: int, mlp_ref: nn.Module):
            def _hook(_module, _args, logits):
                with torch.no_grad():
                    topk_i, topk_w = mlp_ref.route_tokens_to_experts(logits)
                routing[idx] = (topk_i.detach().clone(), topk_w.detach().clone())

            return _hook

        handles.append(mlp.gate.register_forward_hook(_make_hook(i, mlp)))
    return routing, handles


@contextlib.contextmanager
def d9d_routing_replay(
    model_d9d: nn.Module,
    routing: dict[int, tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
) -> Iterator[None]:
    """Monkey-patch `MoELayer.forward` so each instance pulls its (indices,
    weights) from ``routing[layer_idx]`` instead of running its own router.

    HF was run on the full sequence (B, S_hf); d9d runs on the shifted sequence
    (B, S_d9d = S_hf - 1). We reshape HF's flat (B*S_hf, top_k) routing to
    (B, S_hf, top_k), trim to S_d9d along the seq dim, and reflatten — this
    keeps every (batch, position) pair aligned to the same tokens.
    """
    from d9d.module.block.moe.layer import MoELayer

    d9d_id_to_idx: dict[int, int] = {}
    for i in range(len(model_d9d.model.layers)):
        mlp = model_d9d.model.layers[str(i)].mlp
        if isinstance(mlp, MoELayer):
            d9d_id_to_idx[id(mlp)] = i

    original_forward = MoELayer.forward

    def _replay_forward(self, hidden_states):
        layer_idx = d9d_id_to_idx.get(id(self))
        if layer_idx is None or layer_idx not in routing:
            return original_forward(self, hidden_states)

        old_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        if self.shared_expert is not None:
            shared_expert_result = self.shared_expert(hidden_states)
        else:
            shared_expert_result = None

        full_idx, full_w = routing[layer_idx]
        s_d9d = hidden_states.shape[0] // batch_size
        top_k = full_idx.shape[-1]
        expert_indices = full_idx.reshape(batch_size, -1, top_k)[:, :s_d9d, :].reshape(-1, top_k)
        expert_scores = full_w.reshape(batch_size, -1, top_k)[:, :s_d9d, :].reshape(-1, top_k)

        self._update_tokens_per_expert(expert_indices)
        hidden_states, expert_scores, expert_count = self._communicator.dispatch(
            hidden_states, expert_indices, expert_scores
        )
        hidden_states = self.grouped_experts(hidden_states, expert_scores, expert_count)
        hidden_states = self._communicator.combine(hidden_states)

        if shared_expert_result is not None:
            hidden_states = hidden_states + shared_expert_result

        hidden_states = hidden_states.reshape(*old_shape)
        return hidden_states

    MoELayer.forward = _replay_forward
    try:
        yield
    finally:
        MoELayer.forward = original_forward


_ROUTER_GATE_PATTERN = re.compile(r"^(?:model\.)?layers\.\d+\.mlp\.gate\.weight$")


def zero_hf_router_gradients(model_hf: nn.Module) -> None:
    """Zero HF's `mlp.gate.weight` gradients after replay. Pair with
    `d9d_routing_replay` so the gradient comparator's both-zero filter silently
    skips the router gate keys (d9d's are already zero since the router was
    bypassed)."""
    for name, param in model_hf.named_parameters():
        if _ROUTER_GATE_PATTERN.match(name) and param.grad is not None:
            param.grad.zero_()
