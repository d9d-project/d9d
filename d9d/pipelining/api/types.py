from collections.abc import Callable
from typing import Any

import torch

PipelineResultFn = Callable[[dict[str, torch.Tensor], int], Any]
"""
Callback function type for handling results from a final pipeline stage.

Args:
    outputs: A dictionary mapping output names to tensors produced by the stage.
    microbatch_idx: The index of the current micro-batch being processed.

Returns:
    Anything - not used.
"""

PipelineLossFn = Callable[[dict[str, torch.Tensor], int], torch.Tensor]
"""
Callback function type for calculating loss in the final pipeline stage.

Args:
    outputs: A dictionary mapping output names to tensors produced by the model.
    microbatch_idx: The index of the current micro-batch being processed.

Returns:
    The computed loss tensor (scalar).
"""
