"""Defines structural protocols and base classes for PyTorch modules used within the d9d framework."""

from .late_init import ModuleLateInit
from .pipelining import ModuleSupportsPipelining, PipelineStageInfo, distribute_layers_for_pipeline_stage

__all__ = [
    "ModuleLateInit",
    "ModuleSupportsPipelining",
    "PipelineStageInfo",
    "distribute_layers_for_pipeline_stage"
]
