import typing
from collections.abc import Mapping, Sequence
from typing import Protocol

import torch

from d9d.core.types import TensorSpec
from d9d.module.base import ModuleLateInit
from d9d.module.model.io import SequenceInput, SequenceShared, SequenceTransfer
from d9d.pipelining.api import ModuleSupportsPipelining, StageBoundary, TPipelineInput


@typing.runtime_checkable
class TokenEmbeddings(Protocol):
    """Protocol for the backbone's token embedding table."""

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Looks up embeddings for the given token ids.

        Args:
            input_ids: Token ids, shape ``(batch, seq)``.

        Returns:
            Token embeddings, shape ``(batch, seq, hidden)``.
        """
        ...


@typing.runtime_checkable
class DecoderBackbone(
    ModuleLateInit,
    ModuleSupportsPipelining[
        TPipelineInput, SequenceTransfer[torch.Tensor], SequenceShared, SequenceTransfer[torch.Tensor]
    ],
    Protocol[TPipelineInput],
):
    """Protocol for a decoder backbone that maps stage inputs to hidden states.

    A backbone reports its ``hidden_size`` and the split-vocabulary layout it was built with, and
    supports late init and pipelining. This is exactly the surface the task-head composition calls
    on the backbone. The reported dimensions are read-only: they are fixed at construction, and a
    head derives its own shapes from them.

    The pipeline input is a type parameter because it is the one part of the contract a backbone
    family varies: a text-only backbone consumes :class:`SequenceInput`, while a multimodal one
    consumes its own input carrying the media streams its encoders read. Everything downstream of
    the first stage is identical, which is why only this parameter is free.

    Type parameters:
        TPipelineInput: The input consumed on the first stage. Defaults to ``SequenceInput`` for
            the text-only case via the :data:`SequenceDecoderBackbone` alias.
    """

    embed_tokens: TokenEmbeddings
    """The first stage's token embedding table. Only present on the first stage.

    Exposed because a modality encoder composition embeds the tokens itself, in order to merge
    media embeddings into the placeholder positions before the backbone consumes the result.
    """

    @property
    def hidden_size(self) -> int:
        """Dimensionality of the backbone hidden states."""
        ...

    @property
    def split_vocab_size(self) -> Mapping[str, int]:
        """Mapping of vocabulary segment names to their sizes."""
        ...

    @property
    def split_vocab_order(self) -> Sequence[str]:
        """The order in which vocabulary segments are concatenated."""
        ...

    def __call__(
        self, inputs: TPipelineInput | SequenceTransfer[torch.Tensor], shared: SequenceShared
    ) -> SequenceTransfer[torch.Tensor]:
        """Runs the backbone stage as a module, so the composition invokes it through its hooks.

        Args:
            inputs: The pipeline input on the first stage; the incoming ``SequenceTransfer`` otherwise.
            shared: The backbone shared input broadcast to every stage.

        Returns:
            The produced ``SequenceTransfer``.
        """
        ...

    def stage_transfer_spec(
        self, pipeline_input: TPipelineInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        """Describes the ``SequenceTransfer`` crossing the given boundary of this stage.

        Args:
            pipeline_input: A representative pipeline input microbatch; only shapes are read.
            boundary: Which inter-stage edge to describe.

        Returns:
            A ``SequenceTransfer`` of ``TensorSpec``.
        """
        ...


SequenceDecoderBackbone = DecoderBackbone[SequenceInput]
"""A decoder backbone consuming plain token ids — the text-only case."""
