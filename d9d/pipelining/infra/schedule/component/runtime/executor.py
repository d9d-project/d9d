from typing import Any

import torch
from torch.autograd.profiler import record_function

from d9d.core.dist_context import DistributedContext, REGULAR_DOMAIN
from d9d.core.sharding import ShardingSpec, shard_spec_on_dim, shard_tree
from d9d.pipelining.api import PipelineSchedule
from d9d.pipelining.infra.stage import PipelineStage
from .communications import PipelineCommunicationHandler
from .loss import PipelineLossHandler, LossFn
from d9d.pipelining.infra.schedule.component.runtime.action import ActionBase, ActionContext


class PipelineScheduleExecutor(PipelineSchedule):
    """Executes a defined pipeline schedule by interpreting a sequence of actions."""

    def __init__(
            self,
            dist_ctx: DistributedContext,
            stages: list[PipelineStage],
            num_microbatches: int,
            loss_fn: LossFn,

            program: dict[int, list[ActionBase]],
            input_data_sharding_spec: ShardingSpec | None = None,
            input_kwargs_sharding_spec: ShardingSpec | None = None
    ):
        """
        Constructs the schedule executor.

        Args:
            dist_ctx: The distributed context.
            stages: List of stages managed by this executor.
            num_microbatches: Number of microbatches the global batch is split.
            loss_fn: Function to compute loss.
            program: The execution plan mapping rank ID to a list of actions.
            input_data_sharding_spec: Sharding specification for input tensors.
            input_kwargs_sharding_spec: Sharding specification for kwargs.
        """

        self._dist_ctx = dist_ctx
        self._stages = {stage.info.current_stage: stage for stage in stages}
        self._num_microbatches = num_microbatches
        self._program = program

        self._has_backward = any(any(
            action.has_backward_work for action in sub_program
        ) for sub_program in program.values())

        self._comm_handler = PipelineCommunicationHandler(self._stages)
        self._loss_handler = PipelineLossHandler(loss_fn)

        # these could be late-initialized on configure_buffers \/
        self._input_data_sharding_spec = input_data_sharding_spec
        self._input_kwargs_sharding_spec = input_kwargs_sharding_spec


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
        self._dist_ctx.logger.debug(f'Begin pipeline step')
        pp_group = self._dist_ctx.mesh_for(REGULAR_DOMAIN).get_group('pp')

        for stage in self._stages.values():
            stage.reset()

        # Shard inputs and kwargs to microbatches
        inputs = shard_tree(
            inputs,
            num_shards=self._num_microbatches,
            sharding_spec=self._input_data_sharding_spec,
            enforce_even_split=True
        )
        kwargs = shard_tree(
            kwargs,
            num_shards=self._num_microbatches,
            sharding_spec=self._input_kwargs_sharding_spec,
            enforce_even_split=True
        )

        my_program = self._program[pp_group.rank()]

        for action in my_program:
            with record_function(str(action)):
                self._dist_ctx.logger.debug(f'Running pipeline action {action}')
                action.apply(ActionContext(
                    loss=self._loss_handler,
                    stages=self._stages,
                    communications=self._comm_handler,
                    pipeline_inputs_microbatches=inputs,
                    pipeline_kwargs_microbatches=kwargs
                ))

        self._dist_ctx.logger.debug(f'Waiting for potentially hanging PP send comms')
        self._comm_handler.wait_send_all()  # finalize just in case
        self._dist_ctx.logger.debug(f'End pipeline step')
