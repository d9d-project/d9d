from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import nn
from torch.autograd.graph import GradientEdge, Node

from d9d.core.autograd import GLOBAL_GRAD_CONTEXT, GradDirection


def stage_backward_full(
    outputs: list[torch.Tensor], output_grads: list[torch.Tensor] | None, inputs: list[torch.Tensor]
) -> list[torch.Tensor | None]:
    """
    Performs a standard, full backward pass for a pipeline stage.

    This function computes gradients for the inputs based on the gradients
    received for the outputs.

    Args:
        outputs: The output tensors of the forward pass.
        output_grads: The gradients arriving from the next pipeline stage corresponding
            to `outputs`. If None, assumes scalar output or implied ones.
        inputs: The input tensors to the forward pass for which gradients are required.

    Returns:
        A list of gradients corresponding to the `inputs`. If some input does not require gradient - its result will
            be None.
    """

    with GLOBAL_GRAD_CONTEXT.with_directions(GradDirection.inputs, GradDirection.weight):
        torch.autograd.backward(tensors=outputs, grad_tensors=output_grads)

    input_grads = []
    for input_item in inputs:
        input_grads.append(input_item.grad)
        input_item.grad = None
    return input_grads


@dataclass
class ParamGroup:
    """
    Represents a group of parameters and their dependency intermediates in the autograd graph.

    This structure is used to manage the split backward pass, identifying which
    intermediate nodes in the graph allow gradients to flow to specific sets of parameters.

    Attributes:
        params: Set of autograd Nodes representing the parameters.
        intermediates: List of autograd Nodes serving as entry points for gradients
            flowing to these parameters.
        grads: Storage for captured gradients at the intermediate nodes during
            the input backward phase.
    """

    params: set[Node]
    intermediates: list[Node] | None
    grads: list[torch.Tensor | None] | None = None


def _get_grad_fn_or_grad_acc(t: torch.Tensor) -> Node | None:
    if t.requires_grad and t.grad_fn is None:
        # hack from pytorch codebase to create accumulation op
        viewed_t = t.view_as(t)
        grad_fn = viewed_t.grad_fn
        grad_fn = cast(Node, grad_fn)
        return grad_fn.next_functions[0][0]
    else:
        return t.grad_fn


def _construct_reverse_graph(roots: list[Node]) -> dict[Node, list[Node]]:
    """
    Builds a reverse adjacency list (Input -> Output) via BFS from the roots.

    Standard autograd graphs point from Output -> Input (next_functions).
    This helper provides the reverse mapping to assist in dependency analysis.

    Args:
        roots: The starting nodes for the graph traversal.

    Returns:
        A dictionary mapping a node to a list of its dependent (child) nodes.
    """
    reverse_graph = defaultdict(list)
    valid_roots = {x for x in roots if x is not None}
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
    Computes a closure of nodes reachable from roots in the reverse graph.

    Args:
        roots: Starting nodes.
        target_nodes: Nodes that act as boundaries/targets for the search.
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


def _get_param_groups(
    inputs: list[Node], params: list[Node], reverse_edges_dict: dict[Node, list[Node]]
) -> list[ParamGroup]:
    """
    Clusters parameters based on their dependencies on inputs.

    This function identifies how gradients propagate from inputs through intermediates
    to parameters, grouping them to facilitate split backward execution.

    Args:
        inputs: Gradient functions of the input tensors.
        params: Gradient functions of the parameter tensors.
        reverse_edges_dict: The reverse autograd graph.

    Returns:
        A list of distinct parameter groups.
    """

    inputs_closure, _ = _reverse_closure(inputs, set(), reverse_edges_dict)

    node_to_group_map: dict[Node, dict[str, set[Node]]] = {}

    for param in params:
        _, intersected_inputs = _reverse_closure([param], inputs_closure, reverse_edges_dict)

        current_dict = {"params": {param}, "intermediates": intersected_inputs}

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
            unique_groups.append(
                ParamGroup(params=group_dict["params"], intermediates=list(group_dict["intermediates"]))
            )

    return unique_groups


def _make_capture_hook(group: ParamGroup, idx: int) -> Callable[[torch.Tensor], None]:
    def _hook(grad_in: torch.Tensor):
        # Lazy init gradients list
        if group.grads is None and group.intermediates is not None:
            group.grads = [None] * len(group.intermediates)

        if group.grads is not None:
            group.grads[idx] = grad_in

    return _hook


@dataclass
class BackwardInputResult:
    """
    Container for the results of the input backward phase.

    Attributes:
        input_grads: The gradients computed for the input tensors.
        param_groups: The parameter groups with hooks established to capture
            weight gradients in the subsequent phase.
        grad_ownership_tokens: References to tensors keeping the computation
            graph alive for the weight backward phase.
    """

    input_grads: list[torch.Tensor | None]
    param_groups: list[ParamGroup]
    grad_ownership_tokens: list[Any]


def stage_backward_input(
    outputs: list[torch.Tensor],
    output_grads: list[torch.Tensor] | None,
    inputs: list[torch.Tensor],
    weights: Iterator[nn.Parameter],
) -> BackwardInputResult:
    """
    Performs the first phase of a split backward pass: Input Gradients.

    This function computes the gradients with respect to `inputs` while postponing
    the computation of gradients with respect to `weights`. It analyzes the
    autograd graph to identify intermediate nodes where gradients destined for
    weights split off from the main flow. Hooks are registered at these
    intermediates to capture gradients for the second phase (`stage_backward_weight`).

    Args:
        outputs: The output tensors of the forward pass.
        output_grads: The gradients arriving for the outputs.
        inputs: The input tensors from the forward pass.
        weights: An iterator over the model parameters (weights).

    Returns:
        A result object containing input gradients, prepared parameter groups,
        and ownership tokens to maintain graph validity.
    """

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

    with GLOBAL_GRAD_CONTEXT.with_directions(GradDirection.inputs):
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

    for handle in hook_handles:
        handle.remove()

    return BackwardInputResult(
        input_grads=final_input_grads,
        param_groups=param_groups,
        # TODO(max): we can keep only intermediate ownership tokens to both truncate the
        # TODO(max): graph and do not deallocate C++ stuff
        grad_ownership_tokens=outputs,  # Keep the tensors alive!
    )


def stage_backward_weight(  # noqa: C901
    weights: Iterator[nn.Parameter], param_groups: list[ParamGroup], retain_graph: bool = False
) -> tuple[torch.Tensor | None, ...]:
    """
    Performs the second phase of a split backward pass: Weight Gradients.

    This function consumes the gradients captured in the `ParamGroup`s during
    `stage_backward_input` to compute the final gradients for the model weights.
    It triggers backward passes starting from the intermediate nodes identified previously.

    Args:
        weights: An iterator over the model parameters to extract gradients for.
        param_groups: The list of groups containing captured intermediate gradients.
        retain_graph: Whether to retain the graph after this backward pass.

    Returns:
        A tuple of gradients corresponding to the provided `weights`.
    """

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
            for grads_tuple, intermediate in zip(group.grads, group.intermediates, strict=True):
                if grads_tuple is None:
                    raise ValueError("Trying to do backward_weight with to intermediate grads")
                non_none = [g for g in grads_tuple if g is not None]
                if len(non_none) > 0:
                    valid_edges.append(GradientEdge(intermediate, 0))
                    valid_grad_outputs.append(cast(torch.Tensor, sum(non_none)))

        # Break Cycle: Intermediates
        group.intermediates = None

        if valid_edges:
            inputs_for_backward = []
            for node in group.params:
                if node in grad_acc_to_weight:
                    inputs_for_backward.append(grad_acc_to_weight[node])

            if inputs_for_backward:
                with GLOBAL_GRAD_CONTEXT.with_directions(GradDirection.weight):
                    torch.autograd.backward(
                        tensors=valid_edges,
                        grad_tensors=valid_grad_outputs,
                        retain_graph=retain_graph,
                        inputs=inputs_for_backward,
                    )

        # Break Cycle: Grads
        group.grads = None

    return tuple(w.grad for w in all_weights)
