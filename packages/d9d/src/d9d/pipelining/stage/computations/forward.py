from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Generic, cast

from torch import nn

from d9d.pipelining.api import (
    TPipelineInput,
    TPipelineOutput,
    TSharedInput,
    TStageTransfer,
    as_segments,
)

from ..segments import SegmentChain, forward_chains

if TYPE_CHECKING:
    from d9d.parallel.runtime import ParameterRuntime


@dataclasses.dataclass(slots=True)
class ForwardCache(Generic[TPipelineInput, TStageTransfer, TPipelineOutput]):
    """Inputs and outputs of a forward pass, kept for the backward pass.

    Attributes:
        inputs: The stage's incoming ``StageTransfer`` (or ``PipelineInput`` on the first stage) as a
            PyTree.
        outputs: The stage's produced ``StageTransfer`` (or ``PipelineOutput`` on the last stage) as a
            PyTree.
        chain: The executed segments, or ``None`` if the stage ran as one autograd graph.
    """

    inputs: TPipelineInput | TStageTransfer
    outputs: TStageTransfer | TPipelineOutput
    chain: SegmentChain | None = None


class ForwardComputeHandler(Generic[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
    """Forward pass handler for a pipeline stage module.

    It caches the inputs and outputs of each microbatch for the backward pass.
    """

    def __init__(self, stage_index: int, module: nn.Module, runtime: ParameterRuntime | None = None):
        """Constructs the ``ForwardComputeHandler`` object.

        Args:
            stage_index: Logical index of the stage.
            module: The module that runs this stage.
            runtime: The ``ParameterRuntime`` that owns the unsharded parameters, or ``None`` to run the module as one
                autograd graph.
        """
        self._stage_idx = stage_index
        self._module = module
        self._runtime = runtime
        self._segment_source = as_segments(module)

        self._cache: dict[int, ForwardCache[TPipelineInput, TStageTransfer, TPipelineOutput]] = {}

    def run(self, microbatch_index: int, inputs: TPipelineInput | TStageTransfer, shared: TSharedInput):
        """Executes the module's forward pass.

        Args:
            microbatch_index: Identifier for the current microbatch.
            inputs: The stage's ``PipelineInput`` (first stage) or incoming ``StageTransfer``.
            shared: The ``SharedInput`` passed to every stage.

        Raises:
            RuntimeError: If the module forward pass raises. The original exception is chained.
        """
        if self._runtime is not None:
            # The caller already holds the compute context of this microbatch.
            self.run_window([microbatch_index], [inputs], [shared], [nullcontext()])
            return

        try:
            output = self._module(inputs, shared)
        except Exception as e:
            raise RuntimeError(
                f"The forward pass failed on stage ({self._stage_idx}) for microbatch_index ({microbatch_index})."
            ) from e

        self._cache[microbatch_index] = ForwardCache(inputs=inputs, outputs=output)

    def run_window(
        self,
        microbatches: list[int],
        inputs: Sequence[TPipelineInput | TStageTransfer],
        shared: Sequence[TSharedInput],
        contexts: Sequence[AbstractContextManager[None]],
    ):
        """Runs the forward pass of several microbatches, segment by segment.

        Each segment gets its weights once for all of them. It needs a ``ParameterRuntime``.

        Args:
            microbatches: The microbatch identifiers.
            inputs: The stage input of each microbatch.
            shared: The ``SharedInput`` of each microbatch.
            contexts: The compute context of each microbatch. Each is entered once per segment.

        Raises:
            ValueError: If the stage has no ``ParameterRuntime``.
            RuntimeError: If a forward pass raises. The original exception is chained.
        """
        runtime = self._runtime
        if runtime is None:
            raise ValueError(f"Stage ({self._stage_idx}) has no ParameterRuntime, so it cannot run a window.")
        try:
            chains = [self._new_chain(runtime) for _ in microbatches]
            for chain, chain_inputs, chain_shared, context in zip(chains, inputs, shared, contexts, strict=True):
                with context:
                    chain.start_forward(chain_inputs, chain_shared)
            outputs = forward_chains(chains, runtime, self._stage_idx, contexts)
        except Exception as e:
            raise RuntimeError(
                f"The forward pass failed on stage ({self._stage_idx}) for microbatches ({microbatches})."
            ) from e

        for position, microbatch in enumerate(microbatches):
            output = cast(TStageTransfer | TPipelineOutput, outputs[position])
            self._cache[microbatch] = ForwardCache(inputs=inputs[position], outputs=output, chain=chains[position])

    def _new_chain(self, runtime: ParameterRuntime) -> SegmentChain:
        segments = runtime.stage_segments(self._stage_idx)
        return SegmentChain(self._segment_source, segments, self._stage_idx, runtime)

    def get_outputs(self, microbatch_index: int) -> TStageTransfer | TPipelineOutput:
        """Retrieves cached outputs for a specific microbatch without removing them.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            The produced transfer/output PyTree.
        """
        return self._cache[microbatch_index].outputs

    def reset(self):
        """Releases cached forward inputs, outputs and segment graphs."""
        self._cache.clear()

    def pop_cache(self, microbatch_index: int) -> ForwardCache[TPipelineInput, TStageTransfer, TPipelineOutput]:
        """Retrieves and removes the cached forward pass of a microbatch.

        The backward pass calls it.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            The inputs, the outputs and the executed segments of the forward pass.
        """
        return self._cache.pop(microbatch_index)
