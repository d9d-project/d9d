import typing
from collections.abc import Mapping, Sequence
from typing import Protocol

import torch

from d9d.core.types import TensorSpec
from d9d.module.base import ModuleLateInit
from d9d.module.model.io import SequenceInput, SequenceShared, SequenceTransfer
from d9d.pipelining.api import ModuleSupportsPipelining, StageBoundary


@typing.runtime_checkable
class DecoderBackbone(
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequenceShared, SequenceTransfer[torch.Tensor]
    ],
    Protocol,
):
    """Protocol for a decoder backbone that maps stage inputs to hidden states.

    A backbone reports its ``hidden_size`` and its split-vocabulary layout, and supports late init and
    pipelining. The decoder compositions call only this surface. The reported dimensions are fixed at
    construction, and task heads derive their shapes from them.
    """

    @property
    def hidden_size(self) -> int:
        """The size of the backbone hidden states."""
        ...

    @property
    def split_vocab_size(self) -> Mapping[str, int]:
        """The mapping of vocabulary segment names to their sizes."""
        ...

    @property
    def split_vocab_order(self) -> Sequence[str]:
        """The order in which vocabulary segments are concatenated."""
        ...

    def __call__(
        self, inputs: SequenceInput | SequenceTransfer[torch.Tensor], shared: SequenceShared
    ) -> SequenceTransfer[torch.Tensor]:
        """Runs the backbone stage through ``nn.Module.__call__``, so module hooks apply.

        Args:
            inputs: ``SequenceInput`` on the first stage; the incoming ``SequenceTransfer`` otherwise.
            shared: The backbone shared input broadcast to every stage.

        Returns:
            The produced ``SequenceTransfer``.
        """
        ...

    def stage_transfer_spec(
        self, pipeline_input: SequenceInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        """Describes the ``SequenceTransfer`` crossing the given boundary of this stage.

        Args:
            pipeline_input: A representative ``SequenceInput`` microbatch. Only its shapes are read.
            boundary: Which inter-stage edge to describe.

        Returns:
            A ``SequenceTransfer`` of ``TensorSpec``.
        """
        ...
