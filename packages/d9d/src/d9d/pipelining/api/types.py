import abc
from contextlib import AbstractContextManager, nullcontext
from enum import StrEnum
from typing import Generic, TypeVar

import torch

TPipelineInput = TypeVar("TPipelineInput")
TStageTransfer = TypeVar("TStageTransfer")
TSharedInput = TypeVar("TSharedInput")
TPipelineOutput = TypeVar("TPipelineOutput")


class PipelineComputeKind(StrEnum):
    """Kinds of model compute that a pipeline schedule runs for one stage and one microbatch.

    Attributes:
        forward: The forward pass.
        backward: The backward pass for the inputs, and for the weights unless the schedule defers them.
        backward_weight: The deferred backward pass for the weights.
    """

    forward = "forward"
    backward = "backward"
    backward_weight = "backward_weight"


class PipelineStepHandler(abc.ABC, Generic[TPipelineOutput]):
    """Abstract base class for the code that a pipeline step runs around its model compute and on its outputs."""

    def compute_context(
        self, kind: PipelineComputeKind, stage_idx: int, microbatch_idx: int
    ) -> AbstractContextManager[None]:
        """Returns the context to open around one model compute of a stage on a microbatch.

        The context covers the stage module and its autograd graph, but not ``process_outputs()`` and not the
        communications. The default context does nothing.

        A schedule can enter the returned context several times for one compute, e.g. once per region when it runs
        the regions of several microbatches in turn. The context must therefore be reusable, and each entry must
        continue where the previous one stopped, as the forks of ``d9d.rng`` do. A generator from
        ``contextlib.contextmanager`` can be entered only once, so it does not fit.

        Args:
            kind: The kind of the compute.
            stage_idx: The global index of the pipeline stage.
            microbatch_idx: The index of the microbatch.

        Returns:
            The context manager to open while the stage computes.
        """
        return nullcontext()

    @abc.abstractmethod
    def process_outputs(self, outputs: TPipelineOutput, microbatch_idx: int) -> torch.Tensor | None:
        """Handles the ``PipelineOutput`` of the last stage for one microbatch.

        Args:
            outputs: The ``PipelineOutput`` produced by the last stage.
            microbatch_idx: The index of the microbatch.

        Returns:
            The loss of the microbatch if the schedule runs the backward pass, otherwise ``None``.
        """
        ...
