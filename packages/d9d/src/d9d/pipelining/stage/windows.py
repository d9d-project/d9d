from __future__ import annotations

import dataclasses
from collections.abc import Coroutine, Iterable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from d9d import pytree
from d9d.autograd.split_backward import (
    PendingWeightBackward,
    stage_backward_full,
    stage_backward_input,
    stage_backward_weight,
)
from d9d.pipelining.api import ModuleSupportsWindows, WindowBoundary, WindowSteps

if TYPE_CHECKING:
    from d9d.parallel.runtime import ParameterRuntime, ReductionPlan


@dataclasses.dataclass(slots=True)
class _BoundaryTensor:
    """A tensor with a gradient that a region passes to later regions.

    Attributes:
        output: The tensor in the autograd graph of the region that made it.
        input: Its detached copy, which later regions read. The gradient of ``input`` seeds the backward pass of
            ``output``.
    """

    output: torch.Tensor
    input: torch.Tensor


@dataclasses.dataclass(slots=True)
class RegionSeed:
    """The tensors that start the backward pass of one region.

    Attributes:
        outputs: The outputs to differentiate.
        output_grads: Their gradients, or ``None`` for implicit unit gradients.
        inputs: The input leaves whose gradients the chain collects.
    """

    outputs: list[torch.Tensor]
    output_grads: list[torch.Tensor] | None
    inputs: list[torch.Tensor]


@dataclasses.dataclass(slots=True)
class _ExecutedRegion:
    """One region that ran the forward pass of one microbatch.

    Attributes:
        index: The index of the region in the stage.
        group: The parameter group, or ``None`` for the output graph.
        inputs: The input leaves with a gradient that the region read.
        boundaries: The tensors that the region passed to later regions.
        weight_work: The backward work that the weight phase of a split backward pass still has to do: the state
            of the input backward, or the whole seed of a region without inputs. ``None`` when nothing is left.
    """

    index: int
    group: int | None
    inputs: list[torch.Tensor]
    boundaries: list[_BoundaryTensor]
    weight_work: PendingWeightBackward | RegionSeed | None = None


class WindowChain:
    """The regions of one microbatch of one stage, with the autograd graph cut between them.

    Each boundary tensor has one producer. Every consumer adds to its gradient, and consumers run before the
    producer in the backward pass, so the gradient is complete when the producer needs it.

    ``forward_chains`` and ``backward_chains`` run several chains together, region by region. The split backward
    pass of Zero Bubble schedules runs one chain with ``backward_input`` and ``backward_weight``.
    """

    def __init__(self, module: nn.Module, stage_index: int, runtime: ParameterRuntime):
        """Constructs one resumable execution and its backward graph."""
        self._module = module
        self._stage_index = stage_index
        self._runtime = runtime
        self._executed: list[_ExecutedRegion] = []
        self._coroutine: Coroutine[Any, Any, Any] | None = None
        self._pending: WindowBoundary[Any] | None = None
        self._outputs: Any = None
        self._known: set[int] = set()
        # Every tensor with a gradient that a region can read: the stage inputs and the cut copies so far. A region
        # can read a copy from a local variable of the coroutine, not only from the state of its boundary.
        self._leaves: list[torch.Tensor] = []
        self._seed_outputs: list[torch.Tensor] = []
        self._seed_grads: list[torch.Tensor] | None = None
        self._grads: dict[int, torch.Tensor] = {}

    @property
    def finished(self) -> bool:
        """Whether the forward coroutine has returned."""
        return self._coroutine is None

    @property
    def num_regions(self) -> int:
        """The number of executed regions, including the output graph."""
        return len(self._executed)

    @property
    def order(self) -> tuple[int | None, ...]:
        """The parameter group of each executed region."""
        return tuple(region.group for region in self._executed)

    @property
    def pending_group(self) -> int | None:
        """The requested group, or None when forward has completed."""
        if self._pending is None:
            return None
        return self._runtime.group_index(self._stage_index, self._pending.module)

    @property
    def prefetch_group(self) -> int | None:
        """The explicitly requested next group, or ``None`` for the learned order."""
        if self._pending is None or self._pending.prefetch is None:
            return None
        return self._runtime.group_index(self._stage_index, self._pending.prefetch)

    async def _forward(self, inputs: Any, shared: Any) -> Any:
        steps = WindowSteps()
        if isinstance(self._module, ModuleSupportsWindows):
            return await self._module.forward_steps(steps, inputs, shared)
        return await steps.call(self._module, inputs, shared)

    def start_forward(self, inputs: Any, shared: Any):
        """Creates the coroutine and requests its first parameter group."""
        self._known = {id(x) for x in pytree.tree_leaves(inputs) if isinstance(x, torch.Tensor)}
        self._leaves = self._inputs(inputs)
        self._coroutine = self._forward(inputs, shared)
        self._advance(None)
        self._check_boundary_state()

    def _advance(self, state: Any):
        if self._coroutine is None:
            raise ValueError("The forward coroutine is not active.")
        modes = _thread_modes()
        try:
            event = self._coroutine.send(state)
        except StopIteration as completed:
            self._outputs = completed.value
            self._pending = None
            self._coroutine = None
            return
        finally:
            if _thread_modes() != modes:
                raise ValueError(
                    "A context such as no_grad() or autocast spans a boundary of forward_steps(). "
                    "Open and close it between two boundaries."
                )
        if not isinstance(event, WindowBoundary):
            raise TypeError("forward_steps() must yield only WindowBoundary events.")
        self._pending = event

    def _check_boundary_state(self):
        state = self._outputs if self._pending is None else self._pending.state
        if any(id(value) not in self._known for value in self._inputs(state)):
            raise ValueError("Differentiable computation outside a window region is unsupported. Use enter() first.")

    @staticmethod
    def _inputs(state: Any) -> list[torch.Tensor]:
        return list(
            {id(x): x for x in pytree.tree_leaves(state) if isinstance(x, torch.Tensor) and x.requires_grad}.values()
        )

    def forward_region(self):
        """Runs the requested region while its parameter group is available.

        Raises:
            ValueError: If no region is pending.
        """
        if self._pending is None:
            raise ValueError("No region is pending.")
        region = _ExecutedRegion(len(self._executed), self.pending_group, list(self._leaves), [])
        self._executed.append(region)
        self._advance(self._pending.state)
        state = self._outputs if self._pending is None else self._pending.state
        state = self._cut(state, region)
        if self._pending is None:
            self._outputs = state
        else:
            self._pending = WindowBoundary(self._pending.module, state, self._pending.prefetch)

    def next_region(self):
        """Advances past an explicit finish after the previous group's weights are released.

        Raises:
            ValueError: If the forward finishes a region without entering one, or computes outside a region.
        """
        if self._pending is not None and self._pending.module is None:
            self._advance(self._pending.state)
            self._check_boundary_state()
            if self._pending is not None and self._pending.module is None:
                raise ValueError("A finished region must be followed by an entry or a return.")

    def finish_forward(self) -> Any:
        """Adds the output graph that routes loss gradients to boundary producers.

        Returns:
            The final output with its differentiable boundary tensors.
        """
        self._executed.append(_ExecutedRegion(len(self._executed), None, list(self._leaves), []))
        self._leaves = []
        return self._outputs

    def close(self):
        """Closes a suspended forward and drops its local state."""
        if self._coroutine is not None:
            self._coroutine.close()
            self._coroutine = None
        self._pending = None
        self._leaves = []

    def _cut(self, outputs: Any, executed: _ExecutedRegion) -> Any:
        leaves, spec = pytree.tree_flatten(outputs)
        copies: dict[int, torch.Tensor] = {}
        result = []
        for leaf in leaves:
            if not isinstance(leaf, torch.Tensor) or not leaf.requires_grad or id(leaf) in self._known:
                result.append(leaf)
                continue
            if id(leaf) not in copies:
                copies[id(leaf)] = leaf.detach().requires_grad_(requires_grad=True)
                executed.boundaries.append(_BoundaryTensor(output=leaf, input=copies[id(leaf)]))
                self._known.add(id(copies[id(leaf)]))
                self._leaves.append(copies[id(leaf)])
            result.append(copies[id(leaf)])
        return pytree.tree_unflatten(spec, result)

    def start_backward(self, outputs: list[torch.Tensor], output_grads: list[torch.Tensor] | None):
        """Sets what the backward pass of the last region starts from.

        Args:
            outputs: The stage outputs or the loss.
            output_grads: Their gradients, or ``None`` to seed an implicit unit gradient.
        """
        self._seed_outputs = outputs
        self._seed_grads = output_grads
        # The executed regions hold what the backward pass needs.
        self._outputs = None

    def release_region(self, index: int):
        """Drops the references of a region whose backward pass routed its gradients.

        The weight phase of a split backward pass keeps its own state in ``weight_work``.

        Args:
            index: The index of the region.
        """
        executed = self._executed[index]
        executed.inputs = []
        executed.boundaries = []
        if index == len(self._executed) - 1:
            self._seed_outputs = []
            self._seed_grads = None

    def region_seed(self, index: int) -> RegionSeed:
        """Returns what the backward pass of one region starts from.

        Args:
            index: The index of the region.

        Returns:
            The outputs of the region, their gradients and its inputs.
        """
        executed = self._executed[index]
        if index == len(self._executed) - 1:
            return RegionSeed(self._seed_outputs, self._seed_grads, executed.inputs)
        outputs, grads = [], []
        for boundary in executed.boundaries:
            grad = self._grads.pop(id(boundary.input), None)
            # A boundary tensor without a gradient had no differentiable consumer.
            if grad is not None:
                outputs.append(boundary.output)
                grads.append(grad)
        return RegionSeed(outputs, grads, executed.inputs)

    def add_input_grads(self, inputs: Iterable[torch.Tensor], grads: Iterable[torch.Tensor | None]):
        """Adds the gradients of region inputs, which seed the regions that produced them.

        Args:
            inputs: The inputs.
            grads: Their gradients.
        """
        for leaf, grad in zip(inputs, grads, strict=True):
            if grad is None:
                continue
            key = id(leaf)
            self._grads[key] = self._grads[key] + grad if key in self._grads else grad

    def input_grads(self, inputs: list[Any]) -> list[torch.Tensor | None]:
        """Returns the gradients of the stage inputs and lets the chain drop them.

        Args:
            inputs: The leaves of the stage input.

        Returns:
            The gradients, in order. The entry is ``None`` for a leaf without a gradient.
        """
        # The chain lives until the weight phase, but the caller sends the input gradients before it.
        return [self._grads.pop(id(x), None) for x in inputs]

    def _weights(self, executed: _ExecutedRegion) -> list[nn.Parameter]:
        return self._runtime.group_parameters(self._stage_index, executed.group)

    def backward_input(self):
        """Runs the input backward pass of all regions and leaves the weight gradients for ``backward_weight``."""
        for executed in reversed(self._executed):
            with (
                self._runtime.group_weights(self._stage_index, executed.group)
                if executed.group is not None
                else nullcontext()
            ):
                self._backward_input_region(executed)
            self.release_region(executed.index)

    def _backward_input_region(self, executed: _ExecutedRegion):
        # A separate frame, so the seed does not outlive the region.
        seed = self.region_seed(executed.index)
        if not seed.outputs:
            return
        if not executed.inputs:
            # This region has only weight gradients, so its whole graph belongs to the weight phase.
            executed.weight_work = seed
            return
        input_grads, executed.weight_work = stage_backward_input(
            seed.outputs, seed.output_grads, seed.inputs, self._weights(executed)
        )
        self.add_input_grads(executed.inputs, input_grads)

    def backward_weight(self, final: bool | None):
        """Runs the weight backward pass that ``backward_input`` left, and the reductions of the plan.

        Args:
            final: Whether this is the final pass, or ``None`` to skip reductions.
        """
        plan = None if final is None else self._runtime.reduction_plan(self._stage_index, self.order, final)
        for executed in reversed(self._executed):
            prepare = () if plan is None else plan.before_region.get(executed.index, ())
            with (
                self._runtime.group_weights(self._stage_index, executed.group, prepare)
                if executed.group is not None
                else nullcontext()
            ):
                self._backward_weight_region(executed)
            reduce_after_region(self._runtime, plan, executed.index)

    def _backward_weight_region(self, executed: _ExecutedRegion):
        # A separate frame, so the graph of the region does not outlive it.
        work, executed.weight_work = executed.weight_work, None
        match work:
            case RegionSeed():
                stage_backward_full(work.outputs, work.output_grads, inputs=[])
            case PendingWeightBackward():
                stage_backward_weight(work)
            case None:
                # The region had no output with a gradient.
                pass


def _thread_modes() -> tuple[bool, bool, bool, bool]:
    # Grad mode, inference mode and autocast belong to the thread, not to a coroutine. A context that spans a boundary
    # would leak into the regions of the other microbatches.
    return (
        torch.is_grad_enabled(),
        torch.is_inference_mode_enabled(),
        torch.is_autocast_enabled("cuda"),
        torch.is_autocast_enabled("cpu"),
    )


def reduce_after_region(runtime: ParameterRuntime, plan: ReductionPlan | None, region_index: int):
    """Starts the reductions that a plan schedules after one region.

    Args:
        runtime: The runtime that owns the buckets.
        plan: The plan of the backward pass, or ``None`` if the pass reduces nothing.
        region_index: The index of the region that just finished its backward pass.
    """
    if plan is None:
        return
    for bucket_id in plan.after_region.get(region_index, ()):
        runtime.reduce(plan, bucket_id)


def forward_chains(
    chains: Sequence[WindowChain],
    runtime: ParameterRuntime,
    stage_index: int,
    contexts: Sequence[AbstractContextManager[None]],
) -> list[Any]:
    """Runs the forward pass of several microbatches of one stage, region by region.

    Each region gets its weights once for all microbatches.

    Args:
        chains: The chains of the microbatches, after ``start_forward``.
        runtime: The runtime that owns the unsharded parameters.
        stage_index: The index of the stage.
        contexts: The compute context of each microbatch. Each is entered once per region.

    Returns:
        The output of each chain, with boundary tensors linked through the recorded graph.

    Raises:
        ValueError: If microbatches request different groups or the observed order changes.
    """
    position = 0
    declared_next = None
    try:
        while any(not chain.finished for chain in chains):
            groups = [chain.pending_group for chain in chains]
            group = groups[0]
            if group is None or any(item != group for item in groups):
                raise ValueError("Microbatches in a window must request the same parameter groups.")
            if declared_next is not None and group != declared_next:
                raise ValueError("The requested group does not match the explicit prefetch target.")
            expected = runtime.expected_group(stage_index, position)
            if expected is not None and expected != group:
                raise ValueError("The forward changed its parameter group order.")
            hints = [chain.prefetch_group for chain in chains]
            if any(hint != hints[0] for hint in hints):
                raise ValueError("Microbatches in a window must use the same prefetch target.")
            declared_next = hints[0]
            following = (
                declared_next if declared_next is not None else runtime.expected_group(stage_index, position + 1)
            )
            with runtime.group_weights(stage_index, group, next_group=following):
                for chain, context in zip(chains, contexts, strict=True):
                    with context:
                        chain.forward_region()
            for chain, context in zip(chains, contexts, strict=True):
                with context:
                    chain.next_region()
            position += 1
        if declared_next is not None:
            raise ValueError("The forward returned before its explicit prefetch target.")
        runtime.record_order(stage_index, tuple(group for group in chains[0].order if group is not None))
        return [chain.finish_forward() for chain in chains]
    finally:
        for chain, context in zip(chains, contexts, strict=True):
            with context:
                chain.close()


def backward_chains(
    chains: Sequence[WindowChain],
    runtime: ParameterRuntime,
    stage_index: int,
    final: bool | None,
    contexts: Sequence[AbstractContextManager[None]],
):
    """Runs the full backward pass of several microbatches of one stage, region by region.

    Each region gets its weights once for all microbatches. The plan reduces each bucket once, after the region
    where its gradients of all microbatches are complete.

    Args:
        chains: The chains of the microbatches, after ``start_backward``.
        runtime: The runtime that owns the unsharded parameters.
        stage_index: The index of the stage.
        final: Whether this is the final pass, or ``None`` to skip reductions.
        contexts: The compute context of each microbatch. Each is entered once per region.

    Raises:
        ValueError: If the microbatches have different execution traces.
    """
    order = chains[0].order
    if any(chain.order != order for chain in chains):
        raise ValueError("Backward microbatches must have the same execution trace.")
    plan = None if final is None else runtime.reduction_plan(stage_index, order, final)
    for index in reversed(range(len(order))):
        group = order[index]
        following = order[index - 1] if index > 0 else None
        prepare = () if plan is None else plan.before_region.get(index, ())
        scope = nullcontext() if group is None else runtime.group_weights(stage_index, group, prepare, following)
        with scope:
            for chain, context in zip(chains, contexts, strict=True):
                with context:
                    _backward_region(chain, index)
        for chain in chains:
            chain.release_region(index)
        reduce_after_region(runtime, plan, index)


def _backward_region(chain: WindowChain, index: int):
    # A separate frame, so the seed and the results do not outlive the region.
    seed = chain.region_seed(index)
    if not seed.outputs:
        return
    results = stage_backward_full(seed.outputs, seed.output_grads, seed.inputs)
    chain.add_input_grads(seed.inputs, results)
    if index == chain.num_regions - 1:
        # The caller can still hold the loss. Drop its completed graph before earlier regions run.
        for output in seed.outputs:
            if not output.is_leaf and not output._is_view():  # noqa: SLF001 - PyTorch has no public check for views
                output.detach_()
