import dataclasses
import typing

import torch


@dataclasses.dataclass
class PipelineStageInfo:
    """
    Holds information about the current position within the distributed pipeline.

    Attributes:
        current_stage: The 0-based index of the current pipeline stage.
        num_stages: The total number of stages in the pipeline.
    """

    current_stage: int
    num_stages: int

    @property
    def is_current_stage_first(self) -> bool:
        """
        Determines if this is the first stage in the pipeline.

        Returns:
            True if current_stage is 0.
        """

        return self.current_stage == 0

    @property
    def is_current_stage_last(self) -> bool:
        """
        Determines if this is the last stage in the pipeline.

        Returns:
            True if current_stage is the last index.
        """

        return self.current_stage == self.num_stages - 1


def distribute_layers_for_pipeline_stage(
        num_layers: int,
        num_virtual_layers_pre: int,
        num_virtual_layers_post: int,
        stage: PipelineStageInfo
) -> tuple[int, int]:
    """
    Calculates the layer index range for a specific pipeline stage.

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
            raise ValueError(f"Tried to distribute layers, but got {actual_layers} on "
                             f"stage {proposed_stage.current_stage}. Perhaps the pipeline is too long for this model?")

        layer_count_per_stage.append(actual_layers)

    start_layer_id = sum(layer_count_per_stage[:stage.current_stage])
    num_layers_in_stage = layer_count_per_stage[stage.current_stage]

    return start_layer_id, start_layer_id + num_layers_in_stage


@typing.runtime_checkable
class ModuleSupportsPipelining(typing.Protocol):
    """
    Protocol for modules that support pipeline parallelism metadata inference.

    Classes implementing this protocol enable the framework to pre-calculate
    tensor shapes and types required for inter-stage communication (p2p)
    without executing the full forward pass.
    """

    def infer_stage_inputs_from_pipeline_inputs(
            self, inputs: dict[str, torch.Tensor], n_microbatches: int
    ) -> dict[str, torch.Tensor]:
        """
        Infers the input tensors metadata for the current pipeline stage based on global batch inputs.

        Args:
            inputs: Global inputs for the pipeline.
            n_microbatches: Number of microbatches the global batch is split into.

        Returns:
            Dictionary of input tensors expected by this specific stage locally.
        """

        ...

    def infer_stage_outputs_from_pipeline_inputs(
            self, inputs: dict[str, torch.Tensor], n_microbatches: int
    ) -> dict[str, torch.Tensor]:
        """
        Infers the output tensors metadata for the current pipeline stage based on global batch inputs.

        Args:
            inputs: Global inputs for the pipeline (typically a batch).
            n_microbatches: Number of microbatches the global batch is split into.

        Returns:
            Dictionary of output tensors produced by this specific stage locally.
        """

        ...
