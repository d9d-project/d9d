import dataclasses
from typing import Generic

from torch import nn

from d9d.pipelining.api import TPipelineInput, TPipelineOutput, TSharedInput, TStageTransfer


@dataclasses.dataclass(slots=True)
class ForwardCache(Generic[TPipelineInput, TStageTransfer, TPipelineOutput]):
    """Inputs and outputs of a forward pass, kept for the backward pass.

    Attributes:
        inputs: The stage's incoming ``StageTransfer`` (or ``PipelineInput`` on the first stage) as a
            PyTree.
        outputs: The stage's produced ``StageTransfer`` (or ``PipelineOutput`` on the last stage) as a
            PyTree.
    """

    inputs: TPipelineInput | TStageTransfer
    outputs: TStageTransfer | TPipelineOutput


class ForwardComputeHandler(Generic[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
    """Forward pass handler for a pipeline stage module.

    It caches the inputs and outputs of each microbatch for the backward pass.
    """

    def __init__(self, stage_index: int, module: nn.Module):
        """Constructs the ``ForwardComputeHandler`` object.

        Args:
            stage_index: Logical index of the stage.
            module: The module that runs this stage.
        """
        self._stage_idx = stage_index
        self._module = module

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
        try:
            output = self._module(inputs, shared)
        except Exception as e:
            raise RuntimeError(
                f"The forward pass failed on stage ({self._stage_idx}) for microbatch_index ({microbatch_index})."
            ) from e

        self._cache[microbatch_index] = ForwardCache(inputs=inputs, outputs=output)

    def reset(self):
        """Releases cached forward inputs and outputs."""
        self._cache.clear()

    def get_outputs(self, microbatch_index: int) -> TStageTransfer | TPipelineOutput:
        """Retrieves cached outputs for a specific microbatch without removing them.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            The produced transfer/output PyTree.
        """
        return self._cache[microbatch_index].outputs

    def pop_inputs_outputs(
        self, microbatch_index: int
    ) -> tuple[TPipelineInput | TStageTransfer, TStageTransfer | TPipelineOutput]:
        """Retrieves and removes the cached inputs and outputs for a specific microbatch.

        The backward pass calls it.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            A tuple containing (inputs, outputs).
        """
        cache = self._cache.pop(microbatch_index)
        return cache.inputs, cache.outputs
