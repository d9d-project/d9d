import typing
from collections.abc import Mapping
from typing import Generic, Protocol, TypeVar, cast

import torch
from torch import nn

from d9d.core.types import TensorSpec
from d9d.module.base import ModuleLateInit
from d9d.module.block.head.base import TaskHead
from d9d.module.model.io import (
    SequenceHeadsOutput,
    SequenceHeadsShared,
    SequenceInput,
    SequenceShared,
    SequenceTransfer,
)
from d9d.pipelining.api import ModuleSupportsPipelining, PipelineStageInfo, StageBoundary


@typing.runtime_checkable
class DecoderBackbone(
    ModuleLateInit,
    ModuleSupportsPipelining[
        SequenceInput, SequenceTransfer[torch.Tensor], SequenceShared, SequenceTransfer[torch.Tensor]
    ],
    Protocol,
):
    """Protocol for a decoder backbone that maps stage inputs to hidden states.

    A backbone reports its ``hidden_size`` and the split-vocabulary layout it was built with, and
    supports late init and pipelining. This is exactly the surface the task-head composition calls
    on the backbone.

    Attributes:
        hidden_size: Dimensionality of the backbone hidden states.
        split_vocab_size: Mapping of vocabulary segment names to their sizes.
        split_vocab_order: The order in which vocabulary segments are concatenated.
    """

    hidden_size: int
    split_vocab_size: dict[str, int]
    split_vocab_order: list[str]

    def __call__(
        self, inputs: SequenceInput | SequenceTransfer[torch.Tensor], shared: SequenceShared
    ) -> SequenceTransfer[torch.Tensor]:
        """Runs the backbone stage as a module, so the composition invokes it through its hooks.

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
            pipeline_input: A representative ``SequenceInput`` microbatch; only shapes are read.
            boundary: Which inter-stage edge to describe.

        Returns:
            A ``SequenceTransfer`` of ``TensorSpec``.
        """
        ...


TBackbone = TypeVar("TBackbone", bound=DecoderBackbone)


class DecoderWithHeads(
    nn.Module,
    ModuleLateInit,
    ModuleSupportsPipelining[SequenceInput, SequenceTransfer[torch.Tensor], SequenceHeadsShared, SequenceHeadsOutput],
    Generic[TBackbone],
):
    """Composes one decoder backbone with a mapping of prebuilt named task heads.

    The backbone (``self.model``) and the heads (``self.heads``) are public, so a provider
    parallelizes and checkpoint-maps each independently. It does not build heads — that is the
    :func:`build_head` factory's job — so the class stays a pure composition of prebuilt modules
    and lets custom heads in on equal footing. Heads are attached only on the last pipeline stage.
    """

    def __init__(self, backbone: TBackbone, heads: Mapping[str, TaskHead], stage: PipelineStageInfo):
        """Constructs the DecoderWithHeads object.

        Args:
            backbone: The decoder backbone, exposed as ``self.model`` (FQN ``model.*``).
            heads: A mapping of head names to prebuilt task heads, attached on the last stage
                as ``self.heads`` (FQN ``heads.<name>.*``).
            stage: Pipeline stage information for this instance.
        """
        super().__init__()

        self.model = backbone
        self._stage = stage

        if stage.is_current_stage_last:
            self.heads: Mapping[str, TaskHead] = cast(Mapping[str, TaskHead], nn.ModuleDict(dict(heads)))

    def forward(
        self,
        inputs: SequenceInput | SequenceTransfer[torch.Tensor],
        shared: SequenceHeadsShared,
    ) -> SequenceTransfer[torch.Tensor] | SequenceHeadsOutput:
        """Executes the backbone and, on the last stage, every attached head.

        ``shared.sequence`` flows to the backbone; each head receives its own entry of
        ``shared.heads`` under the name it was composed with.

        Args:
            inputs: ``SequenceInput`` on the first stage; the incoming ``SequenceTransfer`` otherwise.
            shared: The backbone shared input plus the per-head shared inputs.

        Returns:
            The produced ``SequenceTransfer`` on non-last stages, or each head's output keyed by
                head name on the last stage.
        """
        model_outputs = self.model(inputs, shared.sequence)

        if not self._stage.is_current_stage_last:
            return model_outputs

        return {name: head(model_outputs.hidden_states, shared.heads[name]) for name, head in self.heads.items()}

    def reset_parameters(self) -> None:
        """Resets module parameters, delegating to the backbone and every head."""
        self.model.reset_parameters()

        if self._stage.is_current_stage_last:
            for head in self.heads.values():
                head.reset_parameters()

    def stage_transfer_spec(
        self, pipeline_input: SequenceInput, boundary: StageBoundary
    ) -> SequenceTransfer[TensorSpec]:
        """Describes the ``SequenceTransfer`` crossing the given boundary, as the backbone does.

        Heads only run on the last stage, whose outgoing edge never transfers, so the transfer is
        entirely the backbone's.

        Args:
            pipeline_input: A representative ``SequenceInput`` microbatch; only shapes are read.
            boundary: Which inter-stage edge to describe.

        Returns:
            A ``SequenceTransfer`` of ``TensorSpec``.
        """
        return self.model.stage_transfer_spec(pipeline_input, boundary)
