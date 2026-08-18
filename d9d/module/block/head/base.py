import abc
from typing import Generic, TypeVar

import torch
from torch import nn

from d9d.module.base import ModuleLateInit

THeadShared = TypeVar("THeadShared")
THeadOutput = TypeVar("THeadOutput")


class TaskHead(nn.Module, ModuleLateInit, abc.ABC, Generic[THeadShared, THeadOutput]):
    """Abstract base class for a task head that turns backbone hidden states into a typed output.

    A head is confined to *compute*: it reads the shared ``hidden_states`` plus its own shared
    input PyTree, and returns its own output PyTree. Its parallelization and checkpoint mapping
    are kept out of the module as separate concerns.

    Type parameters:
        THeadShared: The head's own shared input, routed to it by name at composition time.
        THeadOutput: The output PyTree this head produces.
    """

    @abc.abstractmethod
    def forward(self, hidden_states: torch.Tensor, shared: THeadShared) -> THeadOutput:
        """Computes the head output from hidden states and this head's shared input.

        Args:
            hidden_states: Backbone hidden states of shape ``(B, S, H)``.
            shared: The head's own shared input (e.g. labels, a pooling mask).

        Returns:
            The head's output PyTree.
        """
        ...
