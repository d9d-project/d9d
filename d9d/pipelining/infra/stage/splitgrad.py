from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn
from torch.autograd.graph import Node, GradientEdge


def stage_backward_full(
        outputs: list[torch.Tensor],
        output_grads: list[torch.Tensor],
        inputs: list[torch.Tensor]
) -> list[torch.Tensor]:
    torch.autograd.backward(
        tensors=outputs,
        grad_tensors=output_grads
    )

    input_grads = []
    for input_item in inputs:
        input_grads.append(input_item.grad)
        input_item.grad = None
    return input_grads


@dataclass
class ParamGroup:
    """
    Represents a group of model parameters that share dependency paths
    from the stage inputs.

    Attributes:
        params: The set of AccumulateGrad nodes (weights) in this group.
        intermediates: The set of graph nodes connecting inputs to these weights.
                       Set to None after use to break reference cycles.
        grads: Captured gradients flowing through intermediates.
               Set to None after use to free memory.
    """
    params: set[Node]
    intermediates: list[Node] | None
    grads: list[torch.Tensor | None] | None = None


def _get_grad_fn_or_grad_acc(t: torch.Tensor) -> Node | None:
    if t.requires_grad and t.grad_fn is None:
        # hack from pytorch codebase to create accumulation op
        viewed_t = t.view_as(t)
        grad_fn = viewed_t.grad_fn
        return grad_fn.next_functions[0][0]
    else:
        return t.grad_fn


def _construct_reverse_graph(roots: list[Node]) -> dict[Node, list[Node]]:
    """
    Builds a reverse adjacency list (Input -> Output) via BFS from the roots.
    """
    reverse_graph = defaultdict(list)
    valid_roots = set(x for x in roots if x is not None)
    to_visit = deque(valid_roots)
    visited = set(valid_roots)

    while to_visit:
        current_node = to_visit.popleft()
        for parent_node, _ in current_node.next_functions:
            if parent_node is None:
                continue
            reverse_graph[parent_node].append(current_node)
            if parent_node not in visited:
                visited.add(parent_node)
                to_visit.append(parent_node)

    return reverse_graph


def _reverse_closure(
        roots: list[Node], target_nodes: set[Node], reverse_edges_dict: dict[Node, list[Node]]
) -> tuple[set[Node], set[Node]]:
    """
    Returns the reverse closure of the given roots (nodes reachable by following reversed edges).
    Also returns which 'target_nodes' were encountered during traversal.
    """
    closure: set[Node] = set()
    visited_target_nodes = set()
    to_visit = deque()

    for node in roots:
        if node is not None and node not in closure:
            closure.add(node)
            to_visit.append(node)

    while to_visit:
        node = to_visit.popleft()
        reverse_edges = reverse_edges_dict[node]
        for fn in reverse_edges:
            if fn in closure or fn is None:
                continue
            if fn in target_nodes:
                visited_target_nodes.add(fn)
                continue
            closure.add(fn)
            to_visit.append(fn)

    return closure, visited_target_nodes


def _get_param_groups(
        inputs: list[Node], params: list[Node], reverse_edges_dict: dict[Node, list[Node]]
) -> list[ParamGroup]:
    inputs_closure, _ = _reverse_closure(inputs, set(), reverse_edges_dict)

    node_to_group_map: dict[Node, dict[str, set[Node]]] = {}

    for param in params:
        _, intersected_inputs = _reverse_closure(
            [param], inputs_closure, reverse_edges_dict
        )

        current_dict = {
            "params": {param},
            "intermediates": intersected_inputs
        }

        target_dict = None
        for intermediate_node in intersected_inputs:
            if intermediate_node in node_to_group_map:
                target_dict = node_to_group_map[intermediate_node]
                break

        if target_dict is not None:
            target_dict["params"].update(current_dict["params"])
            target_dict["intermediates"].update(current_dict["intermediates"])
            current_dict = target_dict

        for intermediate_node in current_dict["intermediates"]:
            node_to_group_map[intermediate_node] = current_dict

    # Deduplicate and Convert to Dataclass
    unique_groups = []
    seen_ids = set()
    for group_dict in node_to_group_map.values():
        if id(group_dict) not in seen_ids:
            seen_ids.add(id(group_dict))
            unique_groups.append(ParamGroup(
                params=group_dict["params"],
                intermediates=list(group_dict["intermediates"])
            ))

    return unique_groups


def _make_capture_hook(group: ParamGroup, idx: int):
    def _hook(grad_in: torch.Tensor):
        # Lazy init gradients list
        if group.grads is None and group.intermediates is not None:
            group.grads = [None] * len(group.intermediates)

        if group.grads is not None:
            group.grads[idx] = grad_in

    return _hook


def stage_backward_input(
        outputs: list[torch.Tensor],
        output_grads: list[torch.Tensor] | None,
        inputs: list[torch.Tensor],
        weights: Iterator[nn.Parameter],
) -> tuple[tuple[torch.Tensor | None, ...], list[ParamGroup]]:
    outputs_grad_fn = [grad_fn for x in outputs if (grad_fn := _get_grad_fn_or_grad_acc(x)) is not None]
    inputs_grad_fn = [grad_fn for x in inputs if (grad_fn := _get_grad_fn_or_grad_acc(x)) is not None]
    weights_grad_fn = [grad_fn for x in weights if (grad_fn := _get_grad_fn_or_grad_acc(x)) is not None]

    reverse_edges = _construct_reverse_graph(outputs_grad_fn)
    param_groups = _get_param_groups(inputs_grad_fn, weights_grad_fn, reverse_edges)

    hook_handles = []

    for group in param_groups:
        if group.intermediates:
            for i, node in enumerate(group.intermediates):
                hook_handles.append(node.register_prehook(_make_capture_hook(group, i)))

    if output_grads is None:
        output_grads = [torch.ones_like(o) for o in outputs]

    inputs_requiring_grad = [inp for inp in inputs if inp.requires_grad]

    torch.autograd.backward(
        tensors=outputs,
        grad_tensors=output_grads,
        inputs=inputs_requiring_grad,
        retain_graph=True,
    )

    final_input_grads = []

    # 6. Cleanup
    for input_item in inputs:
        final_input_grads.append(input_item.grad)
        input_item.grad = None

    for t in outputs:
        t.detach_()

    for handle in hook_handles:
        handle.remove()

    return tuple(final_input_grads), param_groups


def stage_backward_weight(
        weights: Iterator[nn.Parameter],
        param_groups: list[ParamGroup],
        retain_graph: bool = False
) -> tuple[torch.Tensor | None, ...]:
    """
    Computes gradients for weights using captured state in ParamGroups.
    """
    # 1. Map Nodes -> Tensors
    grad_acc_to_weight = {}
    all_weights = []  # Keep order

    for weight in weights:
        all_weights.append(weight)
        grad_acc = _get_grad_fn_or_grad_acc(weight)
        if grad_acc is not None:
            grad_acc_to_weight[grad_acc] = weight

    for group in param_groups:
        valid_edges = []
        valid_grad_outputs: list[torch.Tensor] = []

        # Ensure we have data
        if group.grads and group.intermediates:
            for grads_tuple, intermediate in zip(group.grads, group.intermediates):
                if isinstance(grads_tuple, (tuple, list)):
                    non_none = [g for g in grads_tuple if g is not None]
                elif grads_tuple is not None:
                    non_none = [grads_tuple]
                else:
                    non_none = []

                if non_none:
                    valid_edges.append(GradientEdge(intermediate, 0))
                    valid_grad_outputs.append(sum(non_none))

        # Break Cycle: Intermediates
        group.intermediates = None

        if valid_edges:
            inputs_for_backward = []
            for node in group.params:
                if node in grad_acc_to_weight:
                    inputs_for_backward.append(grad_acc_to_weight[node])

            if inputs_for_backward:
                torch.autograd.backward(
                    tensors=valid_edges,
                    grad_tensors=valid_grad_outputs,
                    retain_graph=retain_graph,
                    inputs=inputs_for_backward
                )

        # Break Cycle: Grads
        group.grads = None

    return tuple(w.grad for w in all_weights)
