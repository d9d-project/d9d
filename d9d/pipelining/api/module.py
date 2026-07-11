import dataclasses
import enum
import typing

from d9d.core.types import PyTree, TensorSpec

from .types import TPipelineInput, TPipelineOutput, TSharedInput, TStageTransfer


@dataclasses.dataclass
class PipelineStageInfo:
    """Holds information about the current position within the distributed pipeline.

    Attributes:
        current_stage: The 0-based index of the current pipeline stage.
        num_stages: The total number of stages in the pipeline.
    """

    current_stage: int
    num_stages: int

    @property
    def is_current_stage_first(self) -> bool:
        """Determines if this is the first stage in the pipeline.

        Returns:
            True if current_stage is 0.
        """
        return self.current_stage == 0

    @property
    def is_current_stage_last(self) -> bool:
        """Determines if this is the last stage in the pipeline.

        Returns:
            True if current_stage is the last index.
        """
        return self.current_stage == self.num_stages - 1


def distribute_layers_for_pipeline_stage(
    num_layers: int, num_virtual_layers_pre: int, num_virtual_layers_post: int, stage: PipelineStageInfo
) -> tuple[int, int]:
    """Calculates the layer index range for a specific pipeline stage.

    This function distributes a given number of layers across multiple pipeline
    stages as evenly as possible. It accounts for additional, non-layer
    computational load on the first and last stages (e.g., embeddings and the
    LM head) by using the concept of 'virtual layers' to reserve capacity.

    Args:
        num_layers: The total number of primary model layers to be distributed
            (e.g., the transformer blocks).
        num_virtual_layers_pre: The number of 'virtual' layers representing the
            computational cost of modules on the *first* stage, before the main
            layers (e.g., token and positional embeddings).
        num_virtual_layers_post: The number of 'virtual' layers representing the
            computational cost of modules on the *last* stage, after the main
            layers (e.g., the final layer normalization and LM head).
        stage: An object containing total stages and current stage index.

    Returns:
        A tuple (start_index, end_index), representing the slice of layers for
            the given stage. The start_index is inclusive and the end_index is
            exclusive.

    Raises:
        ValueError: If the pipeline configuration results in a stage having zero
            or negative layers assigned (pipeline too long for the model size).
    """
    num_layers_virtual = num_layers + num_virtual_layers_pre + num_virtual_layers_post

    base_layers_per_stage = num_layers_virtual // stage.num_stages
    extra_layers = num_layers_virtual % stage.num_stages

    layer_count_per_stage = []

    for proposed_stage_i in range(stage.num_stages):
        proposed_stage = PipelineStageInfo(num_stages=stage.num_stages, current_stage=proposed_stage_i)
        layers = base_layers_per_stage + 1 if proposed_stage_i < extra_layers else base_layers_per_stage

        adjustment = 0
        if proposed_stage.is_current_stage_first:
            adjustment += num_virtual_layers_pre
        if proposed_stage.is_current_stage_last:
            adjustment += num_virtual_layers_post

        actual_layers = layers - adjustment

        if actual_layers <= 0:
            raise ValueError(
                f"Tried to distribute layers, but got {actual_layers} on "
                f"stage {proposed_stage.current_stage}. Perhaps the pipeline is too long for this model?"
            )

        layer_count_per_stage.append(actual_layers)

    start_layer_id = sum(layer_count_per_stage[: stage.current_stage])
    num_layers_in_stage = layer_count_per_stage[stage.current_stage]

    return start_layer_id, start_layer_id + num_layers_in_stage


class StageBoundary(enum.Enum):
    """Identifies which inter-stage edge of a stage a transfer spec describes.

    Attributes:
        incoming: The ``StageTransfer`` this stage receives from the previous stage.
        outgoing: The ``StageTransfer`` this stage sends to the next stage.
    """

    incoming = "incoming"
    outgoing = "outgoing"


@typing.runtime_checkable
class ModuleSupportsPipelining(typing.Protocol[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
    """Protocol for modules that can be split across pipeline stages.

    A pipelined module carries four distinct IO roles, each an arbitrary PyTree (dataclasses are the
    recommended form). The module knows its position from the ``PipelineStageInfo`` it receives at
    construction and branches on it explicitly.

    Type parameters:
        TPipelineInput: Input consumed by the *first* stage.
        TStageTransfer: Payload transferred between adjacent stages. The outgoing transfer of stage
            ``N`` and the incoming transfer of stage ``N+1`` are the *same* type.
        TSharedInput: Value passed to *every* stage's forward.
        TPipelineOutput: Output produced by the *last* stage.
    """

    def forward(
        self,
        inputs: TPipelineInput | TStageTransfer,
        shared: TSharedInput,
    ) -> TStageTransfer | TPipelineOutput:
        """Runs this stage.

        Args:
            inputs: ``PipelineInput`` on the first stage; ``StageTransfer`` otherwise.
            shared: The value broadcast to every stage.

        Returns:
            ``PipelineOutput`` on the last stage; ``StageTransfer`` otherwise.
        """
        ...

    def stage_transfer_spec(self, pipeline_input: TPipelineInput, boundary: StageBoundary) -> PyTree[TensorSpec] | None:
        """Describes the ``StageTransfer`` crossing the given boundary of this stage.

        The returned PyTree is structurally identical to the ``StageTransfer`` itself, with every
        tensor leaf replaced by its ``TensorSpec``. Shapes are derived by cheap arithmetic on
        ``pipeline_input``; the ``forward`` body is never executed.

        Args:
            pipeline_input: A representative single microbatch of pipeline input. Only shapes and
                dtypes are read; values are never used.
            boundary: ``incoming`` (received from the previous stage) or ``outgoing`` (sent to the
                next stage).

        Returns:
            A PyTree of ``TensorSpec`` for the transfer crossing that boundary, or ``None`` when the
            boundary is terminal (``incoming`` on the first stage, ``outgoing`` on the last stage).
        """
        ...
