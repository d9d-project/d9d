from collections.abc import Callable
from typing import Any, TypeVar

import torch

TPipelineInput = TypeVar("TPipelineInput")
TStageTransfer = TypeVar("TStageTransfer")
TSharedInput = TypeVar("TSharedInput")
TPipelineOutput = TypeVar("TPipelineOutput")


PipelineResultFn = Callable[[TPipelineOutput, int], Any]
"""Callback function type for handling results from a final pipeline stage.

Args:
    outputs: The ``PipelineOutput`` produced by the last stage.
    microbatch_idx: The index of the current micro-batch being processed.

Returns:
    Anything - not used.
"""

PipelineLossFn = Callable[[TPipelineOutput, int], torch.Tensor]
"""Callback function type for calculating loss in the final pipeline stage.

Args:
    outputs: The ``PipelineOutput`` PyTree produced by the last stage.
    microbatch_idx: The index of the current micro-batch being processed.

Returns:
    The computed loss tensor (scalar).
"""
