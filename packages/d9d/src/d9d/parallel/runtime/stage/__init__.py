"""The parameters of one stage: the weights of its groups, the order of its regions and its backward passes."""

from .regions import BackwardPass
from .stage import StageParameters, bind_stage

__all__ = ["BackwardPass", "StageParameters", "bind_stage"]
