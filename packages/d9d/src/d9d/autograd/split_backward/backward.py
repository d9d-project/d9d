from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.autograd.graph import GradientEdge, Node, get_gradient_edge

from .context import GLOBAL_GRAD_CONTEXT, GradDirection


def stage_backward_full(
    outputs: list[torch.Tensor], output_grads: list[torch.Tensor] | None, inputs: list[torch.Tensor]
) -> list[torch.Tensor | None]:
    """Runs a full backward pass for a pipeline stage.

    It computes the gradients of the inputs and accumulates the weight gradients.

    Args:
        outputs: The output tensors of the forward pass.
        output_grads: The gradients for ``outputs`` from the next pipeline stage. If ``None``,
            ``outputs`` must be scalars, and autograd seeds them with an implicit unit gradient.
        inputs: The input tensors of the forward pass to return gradients for.

    Returns:
        The gradients of ``inputs``, in order. The entry is ``None`` for an input that got no
        gradient.
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
    """Group of parameters and their dependency intermediates in the autograd graph.

    The split backward pass uses it to find the intermediate nodes through which gradients flow to
    a set of parameters.

    Attributes:
        params: The gradient accumulator nodes of the parameters.
        intermediates: The nodes through which gradients enter these parameters, or ``None`` after the
            weight backward consumed them.
        grads: The gradients captured at each intermediate node during the input backward. Each entry
            holds one gradient per output of the node.
    """

    params: set[Node]
    intermediates: list[Node] | None
    grads: list[tuple[torch.Tensor | None, ...] | None] | None = None


def _get_grad_fn_or_grad_acc(t: torch.Tensor) -> Node | None:
    if t.requires_grad and t.grad_fn is None:
        # A leaf's AccumulateGrad node is created lazily, and a dummy view creates it. Mirrors
        # _get_grad_fn_or_grad_acc in torch/distributed/pipelining/_backward.py.
        viewed_t = t.view_as(t)
        grad_fn = viewed_t.grad_fn
        grad_fn = cast(Node, grad_fn)
        return grad_fn.next_functions[0][0]
    else:
        return t.grad_fn


def _construct_reverse_graph(roots: list[Node]) -> dict[Node, list[Node]]:
    """Builds a reverse adjacency list (input -> output) by BFS from the roots.

    Autograd graphs point from output to input (``next_functions``). The reverse mapping serves the
    dependency analysis.

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


def _get_param_groups(
    inputs_closure: set[Node], params: list[Node], reverse_edges_dict: dict[Node, list[Node]]
) -> list[ParamGroup]:
    """Clusters parameters based on their dependencies on inputs.

    This function finds how gradients flow from the inputs through intermediates to the parameters.
    Parameters that share intermediates go into one group.

    Args:
        inputs_closure: The nodes that the gradient functions of the input tensors reach in the reverse graph.
        params: Gradient functions of the parameter tensors.
        reverse_edges_dict: The reverse autograd graph.

    Returns:
        A list of distinct parameter groups.
    """
    groups_by_intermediate: dict[Node, dict[str, set[Node]]] = {}

    for param in params:
        _, intersected_inputs = _reverse_closure([param], inputs_closure, reverse_edges_dict)

        merged_group = {"params": {param}, "intermediates": intersected_inputs}

        # A parameter can connect several earlier groups, so it merges all of them into one.
        overlapping_groups: dict[int, dict[str, set[Node]]] = {}
        for node in intersected_inputs:
            group = groups_by_intermediate.get(node)
            if group is not None:
                overlapping_groups[id(group)] = group
        for group in overlapping_groups.values():
            merged_group["params"].update(group["params"])
            merged_group["intermediates"].update(group["intermediates"])

        for intermediate_node in merged_group["intermediates"]:
            groups_by_intermediate[intermediate_node] = merged_group

    # Several intermediates map to the same group, so deduplicate by identity.
    unique_groups = {id(group): group for group in groups_by_intermediate.values()}
    return [
        ParamGroup(params=group["params"], intermediates=list(group["intermediates"]))
        for group in unique_groups.values()
    ]


def _make_capture_hook(group: ParamGroup, idx: int) -> Callable[[tuple[torch.Tensor | None, ...]], None]:
    def _hook(grad_in: tuple[torch.Tensor | None, ...]):
        if group.grads is None and group.intermediates is not None:
            group.grads = [None] * len(group.intermediates)

        if group.grads is not None:
            group.grads[idx] = grad_in

    return _hook


def _make_clamp_hook(
    grads: tuple[torch.Tensor | None, ...],
) -> Callable[[tuple[torch.Tensor | None, ...]], tuple[torch.Tensor | None, ...]]:
    def _hook(grad_in: tuple[torch.Tensor | None, ...]) -> tuple[torch.Tensor | None, ...]:
        return grads

    return _hook


@dataclass(kw_only=True, slots=True)
class PendingWeightBackward:
    """The weight backward pass that ``stage_backward_input`` deferred.

    Pass it to ``stage_backward_weight`` once. It keeps the autograd graph alive until then.

    Attributes:
        weights: The weights to accumulate the gradients into.
        param_groups: The parameter groups with the gradients captured for the weight backward.
        independent_output_grads: The gradient edges and the gradients of the outputs that do not depend on the
            inputs.
        outputs: The outputs of the forward pass.
        done: Whether ``stage_backward_weight`` already ran it.
    """

    weights: list[nn.Parameter]
    param_groups: list[ParamGroup]
    independent_output_grads: list[tuple[GradientEdge, torch.Tensor]]
    outputs: list[torch.Tensor]
    done: bool = False


def stage_backward_input(
    outputs: list[torch.Tensor],
    output_grads: list[torch.Tensor] | None,
    inputs: list[torch.Tensor],
    weights: Iterable[nn.Parameter],
) -> tuple[list[torch.Tensor | None], PendingWeightBackward]:
    """Runs the first phase of a split backward pass: the input gradients.

    This function computes the gradients of ``inputs`` and defers the gradients of ``weights``. It
    captures the gradients at the intermediate nodes where the weight gradients branch off. The
    second phase, ``stage_backward_weight``, replays them.

    Args:
        outputs: The output tensors of the forward pass.
        output_grads: The gradients arriving for the outputs.
        inputs: The input tensors from the forward pass.
        weights: The model parameters (weights).

    Returns:
        The gradients of ``inputs``, in order, and the deferred weight backward pass. The entry is ``None`` for an
        input that got no gradient.
    """
    outputs_grad_fn = [grad_fn for x in outputs if (grad_fn := _get_grad_fn_or_grad_acc(x)) is not None]
    inputs_grad_fn = [grad_fn for x in inputs if (grad_fn := _get_grad_fn_or_grad_acc(x)) is not None]
    weights = list(weights)
    weights_grad_fn = [grad_fn for x in weights if (grad_fn := _get_grad_fn_or_grad_acc(x)) is not None]

    reverse_edges = _construct_reverse_graph(outputs_grad_fn)
    inputs_closure, _ = _reverse_closure(inputs_grad_fn, set(), reverse_edges)
    param_groups = _get_param_groups(inputs_closure, weights_grad_fn, reverse_edges)

    if output_grads is None:
        output_grads = [torch.ones_like(o) for o in outputs]

    independent_output_grads = [
        (get_gradient_edge(output), grad)
        for output, grad in zip(outputs, output_grads, strict=True)
        if output.requires_grad and _get_grad_fn_or_grad_acc(output) not in inputs_closure
    ]

    hook_handles = []

    for group in param_groups:
        if group.intermediates:
            for i, node in enumerate(group.intermediates):
                hook_handles.append(node.register_prehook(_make_capture_hook(group, i)))

    inputs_requiring_grad = [inp for inp in inputs if inp.requires_grad]

    with GLOBAL_GRAD_CONTEXT.with_directions(GradDirection.inputs):
        torch.autograd.backward(
            tensors=outputs,
            grad_tensors=output_grads,
            inputs=inputs_requiring_grad,
            retain_graph=True,
        )

    final_input_grads = []
    for input_item in inputs:
        final_input_grads.append(input_item.grad)
        input_item.grad = None

    for handle in hook_handles:
        handle.remove()

    return final_input_grads, PendingWeightBackward(
        weights=weights,
        param_groups=param_groups,
        independent_output_grads=independent_output_grads,
        outputs=outputs,
    )


def stage_backward_weight(  # noqa: C901 - edge collection and the single backward pass share the clamp hooks
    pending: PendingWeightBackward,
):
    """Runs the second phase of a split backward pass: the weight gradients.

    It accumulates the gradients of the weights that ``stage_backward_input`` deferred.

    Args:
        pending: The weight backward pass from ``stage_backward_input``.

    Raises:
        ValueError: If ``pending`` already ran, or if no gradient was captured for an intermediate node of a group.
    """
    if pending.done:
        raise ValueError("The weight backward pass already ran. Call stage_backward_input() again for a new one.")
    pending.done = True
    independent_output_grads, pending.independent_output_grads = pending.independent_output_grads, []
    pending.outputs = []

    grad_acc_to_weight = {}
    for weight in pending.weights:
        grad_acc = _get_grad_fn_or_grad_acc(weight)
        if grad_acc is not None:
            grad_acc_to_weight[grad_acc] = weight

    valid_edges = []
    valid_grad_outputs: list[torch.Tensor] = []
    captured: list[tuple[Node, tuple[torch.Tensor | None, ...]]] = []
    inputs_for_backward = []

    for group in pending.param_groups:
        if group.grads and group.intermediates:
            for grads_tuple, intermediate in zip(group.grads, group.intermediates, strict=True):
                if grads_tuple is None:
                    raise ValueError(
                        "No gradient was captured for an intermediate node during the input backward, "
                        "so the weight backward cannot run."
                    )
                captured.append((intermediate, grads_tuple))
                # One edge per output: a multi-output node must get each gradient back on its own output.
                for output_nr, grad in enumerate(grads_tuple):
                    if grad is not None:
                        valid_edges.append(GradientEdge(intermediate, output_nr))
                        valid_grad_outputs.append(grad)
            inputs_for_backward.extend(grad_acc_to_weight[node] for node in group.params if node in grad_acc_to_weight)

        # The group is consumed: drop its nodes and captured gradients so they can be freed.
        group.intermediates = None
        group.grads = None

    run_groups = bool(valid_edges) and bool(inputs_for_backward)
    if not run_groups and not independent_output_grads:
        return

    # Run all groups in one backward pass. A path from one intermediate to a weight can cross nodes of
    # another group, so separate passes would free the graph under each other. Such a path also adds
    # gradient to the intermediates it crosses, on top of their replayed captured gradient. A clamp hook
    # pins each intermediate to its captured gradient, so nothing is counted twice.
    clamp_handles = []
    if len(captured) > 1:
        for intermediate, grads_tuple in captured:
            clamp_handles.append(intermediate.register_prehook(_make_clamp_hook(grads_tuple)))
    try:
        if run_groups:
            with GLOBAL_GRAD_CONTEXT.with_directions(GradDirection.weight):
                # The outputs that do not depend on the inputs can share nodes with the groups.
                torch.autograd.backward(
                    tensors=valid_edges,
                    grad_tensors=valid_grad_outputs,
                    inputs=inputs_for_backward,
                    retain_graph=bool(independent_output_grads),
                )
        if independent_output_grads:
            # Such an output reaches no intermediate, but its graph can pass activation gradients between weights.
            with GLOBAL_GRAD_CONTEXT.with_directions(GradDirection.inputs, GradDirection.weight):
                torch.autograd.backward(
                    tensors=[edge for edge, _ in independent_output_grads],
                    grad_tensors=[grad for _, grad in independent_output_grads],
                    inputs=list(grad_acc_to_weight.values()),
                )
    finally:
        for handle in clamp_handles:
            handle.remove()
