from collections.abc import Callable
from typing import Any

import torch
from d9d.pipelining.api import PipelineStepHandler


class FunctionStepHandler(PipelineStepHandler[Any]):
    def __init__(self, process_outputs: Callable[[Any, int], torch.Tensor | None]):
        self._process_outputs = process_outputs

    def process_outputs(self, outputs: Any, microbatch_idx: int) -> torch.Tensor | None:
        return self._process_outputs(outputs, microbatch_idx)
