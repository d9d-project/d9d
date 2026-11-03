from collections import defaultdict, deque
from dataclasses import dataclass
from typing import cast

import torch
from torch.autograd.graph import GradientEdge, Node


@dataclass
class ParamGroup:
    """Group of parameters and their dependency intermediates in the autograd graph.

    The split backward pass uses it to find the intermediate nodes through which gradients flow to
    a set of parameters.

    Attributes:
        params: The gradient accumulator nodes of the parameters.
        intermediates: The nodes through which gradients enter these parameters, or ``None`` after the
            weight backward consumed them.
        grads: The gradients captured at each intermediate node during the input backward. Each entry
            holds one gradient per output of the node.
        seeds: Output edges and gradients of branches independent of the stage inputs.
    """

    params: set[Node]
    intermediates: list[Node] | None
    grads: list[tuple[torch.Tensor | None, ...] | None] | None = None
    seeds: list[tuple[GradientEdge, torch.Tensor]] | None = None


def gradient_node(t: torch.Tensor) -> Node | None:
    """Returns the backward node or leaf accumulator of a tensor."""
    if t.requires_grad and t.grad_fn is None:
        # A leaf's AccumulateGrad node is created lazily, and a dummy view creates it. Mirrors
        # _get_grad_fn_or_grad_acc in torch/distributed/pipelining/_backward.py.
        viewed_t = t.view_as(t)
        grad_fn = viewed_t.grad_fn
        grad_fn = cast(Node, grad_fn)
        return grad_fn.next_functions[0][0]
    else:
        return t.grad_fn


def reverse_graph(roots: list[Node]) -> dict[Node, list[Node]]:
    """Builds a reverse adjacency list (input -> output) by BFS from the roots.

    Autograd graphs point from output to input (``next_functions``). The reverse mapping serves the
    dependency analysis.

    Args:
        roots: The starting nodes for the graph traversal.

    Returns:
        A dictionary mapping a node to a list of its dependent (child) nodes.
    """
    children: dict[Node, list[Node]] = defaultdict(list)
    valid_roots = {x for x in roots if x is not None}
    to_visit = deque(valid_roots)
    visited = set(valid_roots)

    while to_visit:
        current_node = to_visit.popleft()
        for parent_node, _ in current_node.next_functions:
            if parent_node is None:
                continue
            children[parent_node].append(current_node)
            if parent_node not in visited:
                visited.add(parent_node)
                to_visit.append(parent_node)

    return children


def reverse_closure(
    roots: list[Node], target_nodes: set[Node], reverse_edges_dict: dict[Node, list[Node]]
) -> tuple[set[Node], set[Node]]:
    """Computes a closure of nodes reachable from roots in the reverse graph.

    Args:
        roots: The starting nodes.
        target_nodes: The nodes where the search stops. They are recorded but not expanded.
        reverse_edges_dict: The reverse graph adjacency list.

    Returns:
        A tuple containing the set of all closure nodes and the set of visited target nodes.
    """
    closure: set[Node] = set()
    visited_target_nodes = set()
    to_visit: deque[Node] = deque()

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


@dataclass
class _Dependencies:
    params: set[Node]
    intermediates: set[Node]


def group_parameters(
    inputs: list[Node], params: list[Node], reverse_edges_dict: dict[Node, list[Node]]
) -> list[ParamGroup]:
    """Clusters parameters based on their dependencies on inputs.

    This function finds how gradients flow from the inputs through intermediates to the parameters.
    Parameters that share intermediates go into one group.

    Args:
        inputs: Gradient functions of the input tensors.
        params: Gradient functions of the parameter tensors.
        reverse_edges_dict: The reverse autograd graph.

    Returns:
        A list of distinct parameter groups.
    """
    inputs_closure, _ = reverse_closure(inputs, set(), reverse_edges_dict)

    groups_by_node: dict[Node, _Dependencies] = {}

    for param in params:
        _, intersected_inputs = reverse_closure([param], inputs_closure, reverse_edges_dict)

        merged = _Dependencies(params={param}, intermediates=intersected_inputs)

        overlapping_groups: dict[int, _Dependencies] = {}
        for intermediate_node in intersected_inputs:
            if intermediate_node in groups_by_node:
                group = groups_by_node[intermediate_node]
                overlapping_groups[id(group)] = group
        for group in overlapping_groups.values():
            merged.params.update(group.params)
            merged.intermediates.update(group.intermediates)

        for intermediate_node in merged.intermediates:
            groups_by_node[intermediate_node] = merged

    # Several intermediates can share one group.
    unique_groups = {id(group): group for group in groups_by_node.values()}
    return [
        ParamGroup(params=group.params, intermediates=list(group.intermediates)) for group in unique_groups.values()
    ]
