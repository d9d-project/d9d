from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
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
from d9d.pipelining.api import ModuleSupportsSegments, StageSegment

if TYPE_CHECKING:
    from d9d.parallel.runtime import ParameterRuntime, ReductionPlan


@dataclasses.dataclass(slots=True)
class _BoundaryTensor:
    """A tensor with a gradient that a segment passes to later segments.

    Attributes:
        output: The tensor in the autograd graph of the segment that made it.
        input: Its detached copy, which later segments read. The gradient of ``input`` seeds the backward pass of
            ``output``.
    """

    output: torch.Tensor
    input: torch.Tensor


@dataclasses.dataclass(slots=True)
class SegmentSeed:
    """The tensors that start the backward pass of one segment.

    Attributes:
        outputs: The outputs to differentiate.
        output_grads: Their gradients, or ``None`` for implicit unit gradients.
        inputs: The input leaves whose gradients the chain collects.
    """

    outputs: list[torch.Tensor]
    output_grads: list[torch.Tensor] | None
    inputs: list[torch.Tensor]


@dataclasses.dataclass(slots=True)
class _ExecutedSegment:
    """One segment that ran the forward pass of one microbatch.

    Attributes:
        index: The index of the segment in the stage.
        segment: The segment.
        inputs: The input leaves with a gradient that the segment read.
        boundaries: The tensors that the segment passed to later segments.
        weight_work: The backward work that the weight phase of a split backward pass still has to do: the state
            of the input backward, or the whole seed of a segment without inputs. ``None`` when nothing is left.
    """

    index: int
    segment: StageSegment
    inputs: list[torch.Tensor]
    boundaries: list[_BoundaryTensor]
    weight_work: PendingWeightBackward | SegmentSeed | None = None


class SegmentChain:
    """The segments of one microbatch of one stage, with the autograd graph cut between them.

    Each boundary tensor has one producer. Every consumer adds to its gradient, and consumers run before the
    producer in the backward pass, so the gradient is complete when the producer needs it.

    ``forward_chains`` and ``backward_chains`` run several chains together, segment by segment. The split backward
    pass of Zero Bubble schedules runs one chain with ``backward_input`` and ``backward_weight``.
    """

    def __init__(
        self,
        module: ModuleSupportsSegments,
        segments: Sequence[StageSegment],
        stage_index: int,
        runtime: ParameterRuntime,
    ):
        """Constructs the ``SegmentChain`` object.

        Args:
            module: The stage module.
            segments: The segments of the stage, in forward order.
            stage_index: The index of the stage.
            runtime: The ``ParameterRuntime`` that owns the unsharded parameters.
        """
        self._module = module
        self._segments = segments
        self._stage_index = stage_index
        self._runtime = runtime
        self._executed: list[_ExecutedSegment] = []
        self._constants: Any = None
        self._activations: Any = None
        self._seed_outputs: list[torch.Tensor] = []
        self._seed_grads: list[torch.Tensor] | None = None
        # Keyed by tensor identity. The executed segments keep the tensors alive, so an id stays unique.
        self._grads: dict[int, torch.Tensor] = {}

    @property
    def num_segments(self) -> int:
        """The number of segments of the stage."""
        return len(self._segments)

    def start_forward(self, inputs: Any, shared: Any):
        """Prepares the forward pass of the microbatch.

        Args:
            inputs: The stage input.
            shared: The value broadcast to every stage.

        Raises:
            ValueError: If the constants contain a tensor with a gradient.
        """
        constants = self._module.prepare_segment_constants(shared)
        if any(isinstance(x, torch.Tensor) and x.requires_grad for x in pytree.tree_leaves(constants)):
            raise ValueError(
                "The segment constants contain a tensor with a gradient. Pass it through the activations instead."
            )
        self._constants = constants
        self._activations = inputs

    def forward_segment(self, index: int) -> Any:
        """Runs one segment. The caller puts its weights in place and calls the segments in order.

        Args:
            index: The index of the segment.

        Returns:
            The activations after the segment. They are cut from the graph, unless the segment is the last one.
        """
        inputs = self._activations
        outputs = self._segments[index].module(inputs, self._constants)
        input_leaves = [x for x in pytree.tree_leaves(inputs) if isinstance(x, torch.Tensor)]
        executed = _ExecutedSegment(
            index=index,
            segment=self._segments[index],
            inputs=[x for x in input_leaves if x.requires_grad],
            boundaries=[],
        )
        self._executed.append(executed)
        if index < len(self._segments) - 1:
            outputs = self._cut(outputs, executed, passed_through={id(x) for x in input_leaves})
        self._activations = outputs
        return outputs

    @staticmethod
    def _cut(outputs: Any, executed: _ExecutedSegment, passed_through: set[int]) -> Any:
        leaves, spec = pytree.tree_flatten(outputs)
        copies: dict[int, torch.Tensor] = {}
        result = []
        for leaf in leaves:
            # A leaf the segment only passed along keeps its producer, and its gradient stays in place.
            if not isinstance(leaf, torch.Tensor) or not leaf.requires_grad or id(leaf) in passed_through:
                result.append(leaf)
                continue
            # A tensor that appears twice gets one copy, so its gradients add up in one place.
            if id(leaf) not in copies:
                copies[id(leaf)] = leaf.detach().requires_grad_(requires_grad=True)
                executed.boundaries.append(_BoundaryTensor(output=leaf, input=copies[id(leaf)]))
            result.append(copies[id(leaf)])
        return pytree.tree_unflatten(spec, result)

    def start_backward(self, outputs: list[torch.Tensor], output_grads: list[torch.Tensor] | None):
        """Sets what the backward pass of the last segment starts from.

        Args:
            outputs: The stage outputs or the loss.
            output_grads: Their gradients, or ``None`` to seed an implicit unit gradient.
        """
        self._seed_outputs = outputs
        self._seed_grads = output_grads
        # The executed segments hold what the backward pass needs.
        self._activations = None
        self._constants = None

    def release_segment(self, index: int):
        """Drops the references of a segment whose backward pass routed its gradients.

        The weight phase of a split backward pass keeps its own state in ``weight_work``.

        Args:
            index: The index of the segment.
        """
        executed = self._executed[index]
        executed.inputs = []
        executed.boundaries = []
        if index == len(self._executed) - 1:
            self._seed_outputs = []
            self._seed_grads = None

    def segment_seed(self, index: int) -> SegmentSeed:
        """Returns what the backward pass of one segment starts from.

        Args:
            index: The index of the segment.

        Returns:
            The outputs of the segment, their gradients and its inputs.
        """
        executed = self._executed[index]
        if index == len(self._executed) - 1:
            return SegmentSeed(self._seed_outputs, self._seed_grads, executed.inputs)
        outputs, grads = [], []
        for boundary in executed.boundaries:
            grad = self._grads.pop(id(boundary.input), None)
            # A boundary tensor without a gradient had no differentiable consumer.
            if grad is not None:
                outputs.append(boundary.output)
                grads.append(grad)
        return SegmentSeed(outputs, grads, executed.inputs)

    def add_input_grads(self, inputs: Iterable[torch.Tensor], grads: Iterable[torch.Tensor | None]):
        """Adds the gradients of segment inputs, which seed the segments that produced them.

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

    @staticmethod
    def _weights(executed: _ExecutedSegment) -> list[nn.Parameter]:
        return [p for p in executed.segment.module.parameters() if p.requires_grad]

    def backward_input(self):
        """Runs the input backward pass of all segments and leaves the weight gradients for ``backward_weight``."""
        for executed in reversed(self._executed):
            with self._runtime.segment_weights(self._stage_index, executed.index, writes_grads=False):
                self._backward_input_segment(executed)
            self.release_segment(executed.index)

    def _backward_input_segment(self, executed: _ExecutedSegment):
        # A separate frame, so the seed does not outlive the segment.
        seed = self.segment_seed(executed.index)
        if not seed.outputs:
            return
        if not executed.inputs:
            # This segment has only weight gradients, so its whole graph belongs to the weight phase.
            executed.weight_work = seed
            return
        input_grads, executed.weight_work = stage_backward_input(
            seed.outputs, seed.output_grads, seed.inputs, self._weights(executed)
        )
        self.add_input_grads(executed.inputs, input_grads)

    def backward_weight(self, plan: ReductionPlan | None):
        """Runs the weight backward pass that ``backward_input`` left, and the reductions of the plan.

        Args:
            plan: The buckets to reduce after each segment, or ``None`` without a runtime plan.
        """
        for executed in reversed(self._executed):
            with self._runtime.segment_weights(self._stage_index, executed.index, writes_grads=True):
                self._backward_weight_segment(executed)
            reduce_after_segment(self._runtime, plan, executed.index)

    def _backward_weight_segment(self, executed: _ExecutedSegment):
        # A separate frame, so the graph of the segment does not outlive it.
        work, executed.weight_work = executed.weight_work, None
        match work:
            case SegmentSeed():
                stage_backward_full(work.outputs, work.output_grads, inputs=[])
            case PendingWeightBackward():
                stage_backward_weight(work)
            case None:
                # The segment had no output with a gradient.
                pass


def reduce_after_segment(runtime: ParameterRuntime, plan: ReductionPlan | None, segment_index: int):
    """Starts the reductions that a plan schedules after one segment.

    Args:
        runtime: The runtime that owns the buckets.
        plan: The plan of the backward pass, or ``None`` if the pass reduces nothing.
        segment_index: The index of the segment that just finished its backward pass.
    """
    if plan is None:
        return
    for bucket_id in plan.after_segment.get(segment_index, ()):
        runtime.reduce(plan, bucket_id)


def forward_chains(
    chains: Sequence[SegmentChain],
    runtime: ParameterRuntime,
    stage_index: int,
    contexts: Sequence[AbstractContextManager[None]],
) -> list[Any]:
    """Runs the forward pass of several microbatches of one stage, segment by segment.

    Each segment gets its weights once for all microbatches.

    Args:
        chains: The chains of the microbatches, after ``start_forward``.
        runtime: The runtime that owns the unsharded parameters.
        stage_index: The index of the stage.
        contexts: The compute context of each microbatch. Each is entered once per segment.

    Returns:
        The output of each chain, with the graph of its last segment.
    """
    count = chains[0].num_segments
    outputs: list[Any] = [None] * len(chains)
    for index in range(count):
        following = index + 1 if index + 1 < count else None
        with runtime.segment_weights(stage_index, index, writes_grads=False, next_segment=following):
            for position, (chain, context) in enumerate(zip(chains, contexts, strict=True)):
                with context:
                    outputs[position] = chain.forward_segment(index)
    return outputs


def backward_chains(
    chains: Sequence[SegmentChain],
    runtime: ParameterRuntime,
    stage_index: int,
    plan: ReductionPlan | None,
    contexts: Sequence[AbstractContextManager[None]],
):
    """Runs the full backward pass of several microbatches of one stage, segment by segment.

    Each segment gets its weights once for all microbatches. The plan reduces each bucket once, after the segment
    where its gradients of all microbatches are complete.

    Args:
        chains: The chains of the microbatches, after ``start_backward``.
        runtime: The runtime that owns the unsharded parameters.
        stage_index: The index of the stage.
        plan: The buckets to reduce after each segment, or ``None`` if the pass reduces nothing.
        contexts: The compute context of each microbatch. Each is entered once per segment.
    """
    for index in reversed(range(chains[0].num_segments)):
        following = index - 1 if index > 0 else None
        # Every rank must gather and prepare gradients, even when this rank has no consumer of the segment.
        with runtime.segment_weights(stage_index, index, writes_grads=True, next_segment=following):
            # One autograd pass per microbatch, so that each runs inside its own compute context.
            for chain, context in zip(chains, contexts, strict=True):
                with context:
                    _backward_segment(chain, index)
        for chain in chains:
            chain.release_segment(index)
        reduce_after_segment(runtime, plan, index)


def _backward_segment(chain: SegmentChain, index: int):
    # A separate frame, so the seed and the results do not outlive the segment.
    seed = chain.segment_seed(index)
    if not seed.outputs:
        return
    results = stage_backward_full(seed.outputs, seed.output_grads, seed.inputs)
    chain.add_input_grads(seed.inputs, results)
    if index == chain.num_segments - 1:
        # The caller can still hold the loss. Drop its completed graph before earlier segments run.
        for output in seed.outputs:
            if not output.is_leaf and not output._is_view():  # noqa: SLF001 - PyTorch has no public check for views
                output.detach_()
