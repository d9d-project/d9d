from typing import Any

import torch
from torch.autograd.profiler import record_function

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.core.sharding import shard_spec_on_dim, shard_tree
from d9d.pipelining.api import PipelineSchedule, PipelineShardingSpec
from d9d.pipelining.infra.stage import PipelineStage

from .action import ActionBase, ActionContext
from .communications import PipelineCommunicationHandler
from .loss import LossFn, PipelineLossHandler


class PipelineScheduleExecutor(PipelineSchedule):
    """Executes a defined pipeline schedule by interpreting a sequence of actions."""

    def __init__(
            self,
            dist_context: DistributedContext,
            stages: list[PipelineStage],
            num_microbatches: int,
            loss_fn: LossFn | None,
            program: dict[int, list[ActionBase]],
            sharding_spec: PipelineShardingSpec
    ):
        """
        Constructs the schedule executor.

        Args:
            dist_context: The distributed context.
            stages: List of stages managed by this executor.
            num_microbatches: Number of microbatches the global batch is split.
            loss_fn: Function to compute loss.
            program: The execution plan mapping rank ID to a list of actions.
            sharding_spec: Sharding specification for input and output tensors.
        """

        self._dist_ctx = dist_context
        self._stages = {stage.info.current_stage: stage for stage in stages}
        self._num_microbatches = num_microbatches
        self._program = program

        self._has_backward = any(any(
            action.has_backward_work for action in sub_program
        ) for sub_program in program.values())

        self._comm_handler = PipelineCommunicationHandler(self._stages)
        if loss_fn is None:
            self._loss_handler = None
        else:
            self._loss_handler = PipelineLossHandler(loss_fn)

        # these could be late-initialized on configure_buffers \/
        self._input_data_sharding_spec = sharding_spec.input_data
        self._input_kwargs_sharding_spec = sharding_spec.input_kwargs

    def configure_buffers(self, inputs: dict[str, torch.Tensor], kwargs: dict[str, Any]):
        if self._input_data_sharding_spec is None:
            self._input_data_sharding_spec = shard_spec_on_dim(inputs, dim=0)
        if self._input_kwargs_sharding_spec is None:
            self._input_kwargs_sharding_spec = shard_spec_on_dim(kwargs, dim=0)

        for stage in self._stages.values():
            stage.configure_buffers(
                num_microbatches=self._num_microbatches,
                pipeline_inputs=inputs,
                has_backward=self._has_backward
            )

    def step(self, inputs: dict[str, torch.Tensor], kwargs: dict[str, Any]):
        if self._input_data_sharding_spec is None or self._input_kwargs_sharding_spec is None:
            raise ValueError("Please configure sharding specs first")

        self._dist_ctx.logger.debug("Begin pipeline step")
        pp_group = self._dist_ctx.mesh_for(REGULAR_DOMAIN).get_group("pp")

        for stage in self._stages.values():
            stage.reset()

        # Shard inputs and kwargs to microbatches
        inputs_shard = shard_tree(
            inputs,
            num_shards=self._num_microbatches,
            sharding_spec=self._input_data_sharding_spec,
            enforce_even_split=True
        )
        kwargs_shard = shard_tree(
            kwargs,
            num_shards=self._num_microbatches,
            sharding_spec=self._input_kwargs_sharding_spec,
            enforce_even_split=True
        )

        my_program = self._program[pp_group.rank()]

        for action in my_program:
            with record_function(str(action)):
                self._dist_ctx.logger.debug(f"Running pipeline action {action}")
                action.apply(ActionContext(
                    loss=self._loss_handler,
                    stages=self._stages,
                    communications=self._comm_handler,
                    pipeline_inputs_microbatches=inputs_shard,
                    pipeline_kwargs_microbatches=kwargs_shard
                ))

        self._dist_ctx.logger.debug("Waiting for potentially hanging PP send comms")
        self._comm_handler.wait_send_all()  # finalize just in case
        self._dist_ctx.logger.debug("End pipeline step")
