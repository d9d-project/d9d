import dataclasses
from typing import Any

import torch
from torch import nn
# TODO: usage of PyTorch internal API for backward, migrate to own if it will change a lot
# TODO: generally it's a bad practice but i do not want to write my own backward passes :(
from torch.distributed.pipelining._backward import stage_backward, stage_backward_input, stage_backward_weight

from .struct_helper import DictFlattener


# TODO/NOTICE: We WILL NOT disable FSDP's resharding for microbatches since it will modify
# TODO/NOTICE: its behavior in an unexpected way. Perhaps we need better FSDP resharding policy handler?


@dataclasses.dataclass(slots=True)
class ForwardCache:
    """
    Stores the inputs and outputs of a forward pass to be used later in the backward pass.
    """

    inputs: dict[str, torch.Tensor]
    outputs: dict[str, torch.Tensor]


class ForwardComputeHandler:
    """
    Handles the execution of the forward pass for a pipeline stage module.

    Maintains a cache of inputs and outputs indexed by microbatch ID.
    """

    def __init__(
            self,
            stage_index: int,
            module: nn.Module
    ):
        """
        Constructs a ForwardComputeHandler object.

        Args:
            stage_index: Logical index of the stage.
            module: The PyTorch module representing this stage computation.
        """

        self._stage_idx = stage_index
        self._module = module

        self._cache: dict[int, ForwardCache] = {}

    def run(
            self,
            microbatch_index: int,
            inputs: dict[str, torch.Tensor],
            kwargs: dict[str, Any]
    ):
        """
        Executes the module's forward pass.

        Args:
            microbatch_index: Identifier for the current microbatch.
            inputs: Dictionary of input tensors.
            kwargs: Additional keyword arguments for the module.

        Returns:
            The output of the module.

        Raises:
            RuntimeError: If the forward pass implementation fails.
        """

        # Compute forward
        try:
            output = self._module(**inputs, **kwargs)
        except Exception as e:
            raise RuntimeError(f'S{self._stage_idx}B{microbatch_index} failed to run forward') from e

        self._cache[microbatch_index] = ForwardCache(
            inputs=inputs,
            outputs=output
        )

    def get_outputs(self, microbatch_index: int) -> dict[str, torch.Tensor]:
        """
        Retrieves cached outputs for a specific microbatch without removing them.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            Dictionary of output tensors.
        """

        return self._cache[microbatch_index].outputs

    def pop_inputs_outputs(
            self, microbatch_index: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Retrieves and removes the cached inputs and outputs for a specific microbatch.

        Typically called when initiating the backward pass.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            A tuple containing (inputs, outputs).
        """

        cache = self._cache.pop(microbatch_index)
        return cache.inputs, cache.outputs


@dataclasses.dataclass(kw_only=True, slots=True)
class BackwardCacheInputForWeight:
    """
    State preserved after calculating input gradients, pending weight gradient calculation.
    """

    inputs_grad: dict[str, torch.Tensor]
    param_groups: list[dict[str, Any]] | None


@dataclasses.dataclass(kw_only=True, slots=True)
class BackwardCacheInputForFull:
    stage_outputs_or_loss: list[torch.Tensor]
    output_grads: list[torch.Tensor]
    input_values: list[torch.Tensor]


@dataclasses.dataclass(kw_only=True, slots=True)
class BackwardCacheFull:
    """
    State preserved after calculating weight gradients.
    """

    inputs_grad: dict[str, torch.Tensor]


class BackwardComputeHandler:
    """
    Handles the execution of backward passes for a pipeline stage.

    Supports splitting the backward pass into input-gradients and weight-gradients
    phases, which is necessary for schedules like ZB.
    """

    def __init__(
            self,
            stage_index: int,
            module: nn.Module
    ):
        """
        Constructs a BackwardComputeHandler object.

        Args:
            stage_index: Logical index of the stage.
            module: The PyTorch module to compute gradients for.
        """

        self._stage_idx = stage_index
        self._module = module

        self._cache: dict[int, BackwardCacheInputForWeight | BackwardCacheInputForFull | BackwardCacheFull] = {}

    def backward_full(
            self,
            microbatch_index: int,
            inputs: dict[str, torch.Tensor],
            outputs: dict[str, torch.Tensor],
            outputs_grad: dict[str, torch.Tensor] | None,
    ):
        """
        Performs a full backward pass (both inputs and weights).

        Args:
            microbatch_index: Identifier for the microbatch.
            inputs: The inputs used in the forward pass.
            outputs: The outputs produced by the forward pass.
            outputs_grad: Gradients of the loss with respect to the outputs.
        """

        if microbatch_index in self._cache:
            raise ValueError(f'S{self._stage_idx}B{microbatch_index} double backward')

        inputs_flattener = DictFlattener(inputs.keys())
        outputs_flattener = DictFlattener(outputs.keys())

        inputs_grad_linear = stage_backward(
            stage_output=outputs_flattener.flatten(outputs),
            output_grads=outputs_flattener.flatten(outputs_grad),
            input_values=inputs_flattener.flatten(inputs)
        )

        self._cache[microbatch_index] = BackwardCacheFull(
            inputs_grad=inputs_flattener.unflatten(inputs_grad_linear)
        )

    def backward_input(
            self,
            microbatch_index: int,
            inputs: dict[str, torch.Tensor],
            outputs: dict[str, torch.Tensor],
            outputs_grad: dict[str, torch.Tensor] | None,
            is_first_stage: bool
    ):
        """
        Performs a partial backward pass to compute gradients with respect to inputs only.

        This prepares the computation state for a subsequent `backward_weight` call.

        Args:
            microbatch_index: Identifier for the microbatch.
            inputs: The inputs used in the forward pass.
            outputs: The outputs produced by the forward pass.
            outputs_grad: Gradients of the loss with respect to the outputs.
        """

        # TODO: fix accumulation hooks being executed properly
        if microbatch_index in self._cache:
            raise ValueError(f'Double backward pass')

        inputs_flattener = DictFlattener(inputs.keys())
        outputs_flattener = DictFlattener(outputs.keys())

        if is_first_stage:
            self._cache[microbatch_index] = BackwardCacheInputForFull(
                stage_outputs_or_loss=outputs_flattener.flatten(outputs),
                output_grads=outputs_flattener.flatten(outputs_grad),
                input_values=inputs_flattener.flatten(inputs)
            )
        else:
            inputs_grad_linear, param_groups = stage_backward_input(
                stage_outputs_or_loss=outputs_flattener.flatten(outputs),
                output_grads=outputs_flattener.flatten(outputs_grad),
                input_values=inputs_flattener.flatten(inputs),
                weights=self._module.parameters()
            )

            self._cache[microbatch_index] = BackwardCacheInputForWeight(
                inputs_grad=inputs_flattener.unflatten(inputs_grad_linear),
                param_groups=param_groups
            )

    def backward_weight(
            self,
            microbatch_index: int
    ):
        """
        Performs a partial backward pass to accumulate gradients into weights.

        Must be preceded by `backward_input` for the same microbatch index.

        Args:
            microbatch_index: Identifier for the microbatch.
        """

        # TODO: fix accumulation hooks being executed properly
        if microbatch_index not in self._cache:
            raise ValueError(f'S{self._stage_idx}BW{microbatch_index} - weight backward with no input backward before')

        prev_cache = self._cache.pop(microbatch_index)

        match prev_cache:
            case BackwardCacheInputForFull():
                stage_backward(
                    stage_output=prev_cache.stage_outputs_or_loss,
                    output_grads=prev_cache.output_grads,
                    input_values=prev_cache.input_values
                )
            case BackwardCacheInputForWeight():
                stage_backward_weight(
                    weights=self._module.parameters(),
                    param_groups=prev_cache.param_groups
                )
            case _:
                raise ValueError('Previous backward was not input backward')


    def pop_for_sending(self, microbatch_index: int) -> dict[str, torch.Tensor]:
        """
        Retrieves the calculated input gradients for a microbatch.

        Args:
            microbatch_index: Identifier for the microbatch.

        Returns:
            Dictionary of gradient tensors.
        """
        cached = self._cache[microbatch_index]

        # Pop only if it is BW
        if isinstance(cached, BackwardCacheFull):
            del self._cache[microbatch_index]

        return cached.inputs_grad
