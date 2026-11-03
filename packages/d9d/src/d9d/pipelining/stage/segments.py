from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from d9d import pytree
from d9d.autograd.split_backward import (
    BackwardInputResult,
    stage_backward_full,
    stage_backward_input,
    stage_backward_weight,
)
from d9d.pipelining.api import ModuleSupportsSegments, StageSegment

if TYPE_CHECKING:
    from d9d.parallel.runtime import ParameterRuntime, ReductionPlan


def validate_segments(module: nn.Module, segments: Sequence[StageSegment]):
    """Checks that a stage has segments and that they hold every parameter of the module.

    Every parameter needs its unsharded value during forward, including frozen or inference-only weights.

    Args:
        module: The stage module.
        segments: The segments of the stage.

    Raises:
        ValueError: If the stage has no segments, or if a parameter belongs to no segment.
    """
    if not segments:
        raise ValueError(f"The stage module ({type(module).__name__}) has no segments. Return at least one.")
    owned = {id(p) for segment in segments for m in segment.modules for p in m.parameters()}
    missing = [name for name, p in module.named_parameters() if id(p) not in owned]
    if missing:
        raise ValueError(f"The parameters ({missing}) belong to no segment. Add their modules to a segment.")


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
    weight_work: BackwardInputResult | SegmentSeed | None = None


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
        outputs = self._segments[index].call(inputs, self._constants)
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
        """Returns the gradients of the stage inputs.

        Args:
            inputs: The leaves of the stage input.

        Returns:
            The gradients, in order. The entry is ``None`` for a leaf without a gradient.
        """
        return [self._grads.get(id(x)) for x in inputs]

    @staticmethod
    def _weights(executed: _ExecutedSegment) -> list[nn.Parameter]:
        return [p for module in executed.segment.modules for p in module.parameters() if p.requires_grad]

    def backward_input(self):
        """Runs the input backward pass of all segments and leaves the weight gradients for ``backward_weight``."""
        for executed in reversed(self._executed):
            seed = self.segment_seed(executed.index)
            with self._runtime.segment_weights(self._stage_index, executed.index, writes_grads=False):
                if not seed.outputs:
                    continue
                if not executed.inputs:
                    # This segment has only weight gradients, so its whole graph belongs to the weight phase.
                    executed.weight_work = seed
                else:
                    result = stage_backward_input(
                        seed.outputs, seed.output_grads, seed.inputs, iter(self._weights(executed))
                    )
                    executed.weight_work = result
                    self.add_input_grads(executed.inputs, result.input_grads)

    def backward_weight(self, plan: ReductionPlan | None):
        """Runs the weight backward pass that ``backward_input`` left, and the reductions of the plan.

        Args:
            plan: The buckets to reduce after each segment, or ``None`` without a runtime plan.
        """
        for executed in reversed(self._executed):
            with self._runtime.segment_weights(self._stage_index, executed.index, writes_grads=True):
                match executed.weight_work:
                    case SegmentSeed() as seed:
                        stage_backward_full(seed.outputs, seed.output_grads, inputs=[])
                    case BackwardInputResult() as result:
                        stage_backward_weight(iter(self._weights(executed)), result.param_groups)
                    case None:
                        pass
                executed.weight_work = None
            reduce_after_segment(self._runtime, plan, executed.index)


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
        runtime.reduce(bucket_id, plan.final)


def forward_chains(chains: Sequence[SegmentChain], runtime: ParameterRuntime, stage_index: int) -> list[Any]:
    """Runs the forward pass of several microbatches of one stage, segment by segment.

    Each segment gets its weights once for all microbatches.

    Args:
        chains: The chains of the microbatches, after ``start_forward``.
        runtime: The runtime that owns the unsharded parameters.
        stage_index: The index of the stage.

    Returns:
        The output of each chain, with the graph of its last segment.
    """
    count = chains[0].num_segments
    outputs: list[Any] = [None] * len(chains)
    for index in range(count):
        following = index + 1 if index + 1 < count else None
        with runtime.segment_weights(stage_index, index, writes_grads=False, next_segment=following):
            for position, chain in enumerate(chains):
                outputs[position] = chain.forward_segment(index)
    return outputs


def backward_chains(
    chains: Sequence[SegmentChain], runtime: ParameterRuntime, stage_index: int, plan: ReductionPlan | None
):
    """Runs the full backward pass of several microbatches of one stage, segment by segment.

    Each segment gets its weights once and runs one autograd pass for all microbatches. The plan reduces each
    bucket once, after the segment where its gradients of all microbatches are complete.

    Args:
        chains: The chains of the microbatches, after ``start_backward``.
        runtime: The runtime that owns the unsharded parameters.
        stage_index: The index of the stage.
        plan: The buckets to reduce after each segment, or ``None`` if the pass reduces nothing.
    """
    for index in reversed(range(chains[0].num_segments)):
        following = index - 1 if index > 0 else None
        # Every rank must gather and prepare gradients, even when this rank has no consumer of the segment.
        with runtime.segment_weights(stage_index, index, writes_grads=True, next_segment=following):
            seeds = [chain.segment_seed(index) for chain in chains]
            outputs = [x for seed in seeds for x in seed.outputs]
            grads = [x for seed in seeds for x in (seed.output_grads or ())]
            inputs = [x for seed in seeds for x in seed.inputs]
            if outputs:
                # One engine run per segment: starting it costs more than the work of a small segment.
                results = stage_backward_full(outputs, grads or None, inputs)
                position = 0
                for chain, seed in zip(chains, seeds, strict=True):
                    chain.add_input_grads(seed.inputs, results[position : position + len(seed.inputs)])
                    position += len(seed.inputs)
        reduce_after_segment(runtime, plan, index)
