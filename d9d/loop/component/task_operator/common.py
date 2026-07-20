import typing

from d9d.core.types import MicrobatchPack, PyTree
from d9d.loop.control import BaseTask, BuildForwardInputsContext

from ..pipeline_state import PipelineStateHandler

TBatch = typing.TypeVar("TBatch", bound=PyTree)
TPipelineInput = typing.TypeVar("TPipelineInput")
TSharedInput = typing.TypeVar("TSharedInput")
TState = typing.TypeVar("TState", bound=PyTree)


def build_pipeline_microbatch_inputs(
    task: BaseTask[TBatch, TPipelineInput, TSharedInput, TState],
    pipeline_state: PipelineStateHandler[TState],
    pack: MicrobatchPack,
) -> tuple[tuple[TPipelineInput, ...], tuple[TSharedInput, ...]]:
    """Builds the per-microbatch model inputs for a whole pack, storing each microbatch's side-data.

    Args:
        task: The task mapping one raw microbatch into model inputs.
        pipeline_state: The per-microbatch store the returned side-data is written into.
        pack: The step's pack of raw microbatches.

    Returns:
        A tuple of per-microbatch ``PipelineInput`` and a tuple of per-microbatch ``SharedInput``.
    """
    inputs_microbatches = []
    shared_microbatches = []

    for microbatch_idx, microbatch in enumerate(pack):
        model_inputs = task.build_forward_inputs(BuildForwardInputsContext(batch=microbatch))
        pipeline_state.store(microbatch_idx, model_inputs.state)
        inputs_microbatches.append(model_inputs.input)
        shared_microbatches.append(model_inputs.shared)

    return tuple(inputs_microbatches), tuple(shared_microbatches)
