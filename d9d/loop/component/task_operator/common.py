from d9d.core.types import MicrobatchPack
from d9d.loop.control import BaseTask, BuildForwardInputsContext

from ..pipeline_state import PipelineStateHandler


def build_pipeline_microbatch_inputs(
    task: BaseTask, pipeline_state: PipelineStateHandler, pack: MicrobatchPack
) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """Builds the per-microbatch model inputs for a whole pack, storing each microbatch's side-data.

    Args:
        task: The task mapping one raw microbatch into model inputs.
        pipeline_state: The per-microbatch store the returned side-data is written into.
        pack: The step's pack of raw microbatches.

    Returns:
        A tuple of per-microbatch input dicts and a tuple of per-microbatch kwarg dicts.
    """
    inputs_microbatches = []
    kwargs_microbatches = []

    for microbatch_idx, microbatch in enumerate(pack):
        model_inputs = task.build_forward_inputs(BuildForwardInputsContext(batch=microbatch))
        pipeline_state.store(microbatch_idx, model_inputs.state)
        inputs_microbatches.append(model_inputs.inputs)
        kwargs_microbatches.append(model_inputs.kwargs)

    return tuple(inputs_microbatches), tuple(kwargs_microbatches)
