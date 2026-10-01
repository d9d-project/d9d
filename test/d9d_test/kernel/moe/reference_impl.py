import torch


def routing_order_torch(routing_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # permuted rows are grouped by expert, tokens keep their order within an expert
    expert_ids, token_ids = routing_map.T.nonzero(as_tuple=True)
    return token_ids, expert_ids


def permute_torch(x: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids, expert_ids = routing_order_torch(routing_map)
    return x[token_ids], probs[token_ids, expert_ids]


def unpermute_torch(y: torch.Tensor, routing_map: torch.Tensor, merging_probs: torch.Tensor | None) -> torch.Tensor:
    token_ids, expert_ids = routing_order_torch(routing_map)
    if merging_probs is not None:
        y = y * merging_probs[token_ids, expert_ids][:, None]
    out = torch.zeros((routing_map.shape[0], y.shape[1]), dtype=y.dtype, device=y.device)
    return out.index_add(0, token_ids, y)
