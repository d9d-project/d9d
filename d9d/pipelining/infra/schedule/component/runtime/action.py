import abc
import dataclasses
from enum import StrEnum
from typing import Any

import torch

from d9d.pipelining.infra.stage import PipelineStage

from .communications import PipelineCommunicationHandler
from .loss import PipelineLossHandler


@dataclasses.dataclass(kw_only=True, slots=True)
class ActionContext:
    """
    Holds the runtime context required to execute a pipeline action.

    Attributes:
        pipeline_inputs_microbatches: The global inputs sharded by microbatch.
        pipeline_kwargs_microbatches: The global keyword arguments sharded by microbatch.
        stages: A mapping of stage indices to their active PipelineStage instances.
        communications: The handler for P2P communications.
        loss: The handler for loss computation, or None if not available.
    """

    pipeline_inputs_microbatches: tuple[dict[str, torch.Tensor], ...]
    pipeline_kwargs_microbatches: tuple[dict[str, Any], ...]

    stages: dict[int, PipelineStage]
    communications: PipelineCommunicationHandler
    loss: PipelineLossHandler | None


class ActionWorkType(StrEnum):
    """
    Classifies the type of work performed by an action.

    Attributes:
        compute: Indicates the action involves computation components (forward, backward).
        communicate: Indicates the action involves network I/O components (send, receive).
    """

    compute = "compute"
    communicate = "communicate"


class ActionBase(abc.ABC):
    """
    Abstract base class for all pipeline schedule actions.

    An action represents an atomic unit of work in a pipeline schedule,
    such as computing a microbatch or sending/receiving a tensor.
    """

    @abc.abstractmethod
    def apply(self, ctx: ActionContext):
        """
        Executes the action logic using the provided context.

        Args:
            ctx: The runtime context containing stages, data, and communication handlers.
        """

        ...

    @property
    @abc.abstractmethod
    def work_type(self) -> ActionWorkType:
        """Returns the classification of work this action performs."""
        ...

    @property
    @abc.abstractmethod
    def has_backward_work(self) -> bool:
        """Returns True if this action involves backward pass computations."""
        ...

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns a short string representation of the action for logging/visualization."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardSendAction(ActionBase):
    """
    Action to schedule a forward pass tensor send operation.

    Attributes:
        stage_idx: The integer index of the pipeline stage initiating the send operation.
        microbatch_idx: The integer index of the microbatch being sent.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext):
        ctx.communications.schedule_fwd_send(self.stage_idx, self.microbatch_idx)

    @property
    def work_type(self) -> ActionWorkType:
        return ActionWorkType.communicate

    @property
    def has_backward_work(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"{self.stage_idx}SEND_F{self.microbatch_idx}"


@dataclasses.dataclass(frozen=True, slots=True)
class BackwardSendAction(ActionBase):
    """
    Action to schedule a backward pass gradient send operation.

    Attributes:
        stage_idx: The integer index of the pipeline stage initiating the send operation.
        microbatch_idx: The integer index of the microbatch being sent.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext):
        ctx.communications.schedule_bwd_send(self.stage_idx, self.microbatch_idx)

    @property
    def work_type(self) -> ActionWorkType:
        return ActionWorkType.communicate

    @property
    def has_backward_work(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"{self.stage_idx}SEND_B{self.microbatch_idx}"


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardReceiveAction(ActionBase):
    """
    Action to schedule a forward pass tensor receive operation.

    Attributes:
        stage_idx: The integer index of the pipeline stage expecting the receive operation.
        microbatch_idx: The integer index of the microbatch being received.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext):
        ctx.communications.schedule_fwd_recv(self.stage_idx, self.microbatch_idx)

    @property
    def work_type(self) -> ActionWorkType:
        return ActionWorkType.communicate

    @property
    def has_backward_work(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"{self.stage_idx}RECV_F{self.microbatch_idx}"


@dataclasses.dataclass(frozen=True, slots=True)
class BackwardReceiveAction(ActionBase):
    """
    Action to schedule a backward pass gradient receive operation.

    Attributes:
        stage_idx: The integer index of the pipeline stage expecting the receive operation.
        microbatch_idx: The integer index of the microbatch being received.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext):
        ctx.communications.schedule_bwd_recv(self.stage_idx, self.microbatch_idx)

    @property
    def work_type(self) -> ActionWorkType:
        return ActionWorkType.communicate

    @property
    def has_backward_work(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"{self.stage_idx}RECV_B{self.microbatch_idx}"


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardComputeAction(ActionBase):
    """
    Action to perform forward computation for a specific microbatch.

    Attributes:
        stage_idx: The integer index of the pipeline stage.
        microbatch_idx: The integer index of the microbatch to compute.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext):
        # todo check unsharded
        stage = ctx.stages[self.stage_idx]

        if not stage.info.is_current_stage_first and self.stage_idx - 1 not in ctx.stages:
            ctx.communications.wait_fwd_recv(self.stage_idx, self.microbatch_idx)

        stage.forward_one_chunk(
            microbatch_index=self.microbatch_idx,
            pipeline_inputs=ctx.pipeline_inputs_microbatches[self.microbatch_idx],
            pipeline_kwargs=ctx.pipeline_kwargs_microbatches[self.microbatch_idx]
        )
        result = stage.get_local_fwd_output(self.microbatch_idx)

        if stage.info.is_current_stage_last and ctx.loss is not None:
            ctx.loss.compute_loss(result, self.microbatch_idx)

        if not stage.info.is_current_stage_last and self.stage_idx + 1 in ctx.stages:
            ctx.stages[self.stage_idx + 1].set_local_fwd_input(
                inputs=result,
                microbatch_index=self.microbatch_idx
            )

    @property
    def work_type(self) -> ActionWorkType:
        return ActionWorkType.compute

    @property
    def has_backward_work(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"{self.stage_idx}F{self.microbatch_idx}"


@dataclasses.dataclass(frozen=True, slots=True)
class BackwardFullInputComputeAction(ActionBase):
    """
    Action to perform backward computation with respect to inputs.

    Attributes:
        stage_idx: The integer index of the pipeline stage.
        microbatch_idx: The integer index of the microbatch to compute.
        full_backward: If True, performs a full backward pass including inputs
            and weights. If False, may only compute gradients w.r.t inputs
            (depending on schedule implementation).
    """

    stage_idx: int
    microbatch_idx: int
    full_backward: bool

    def apply(self, ctx: ActionContext):
        # todo unshard
        stage = ctx.stages[self.stage_idx]

        if not stage.info.is_current_stage_last and self.stage_idx + 1 not in ctx.stages:
            ctx.communications.wait_bwd_recv(self.stage_idx, self.microbatch_idx)

        if stage.info.is_current_stage_last and ctx.loss is not None:
            loss = ctx.loss.acquire_loss(self.microbatch_idx)
        else:
            loss = None

        stage.backward_one_chunk(
            microbatch_index=self.microbatch_idx,
            full_backward=self.full_backward,
            loss=loss
        )

        if not stage.info.is_current_stage_first and self.stage_idx - 1 in ctx.stages:
            ctx.stages[self.stage_idx - 1].set_local_bwd_input(
                microbatch_index=self.microbatch_idx,
                inputs=stage.pop_local_bwd_output(self.microbatch_idx)
            )

    @property
    def work_type(self) -> ActionWorkType:
        return ActionWorkType.compute

    @property
    def has_backward_work(self) -> bool:
        return True

    def __str__(self) -> str:
        letter = "B" if self.full_backward else "I"
        return f"{self.stage_idx}{letter}{self.microbatch_idx}"


@dataclasses.dataclass(frozen=True, slots=True)
class BackwardWeightComputeAction(ActionBase):
    """
    Action to perform gradient accumulation on weights.

    Attributes:
        stage_idx: The integer index of the pipeline stage.
        microbatch_idx: The integer index of the microbatch to compute.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext):
        # todo unshard
        stage = ctx.stages[self.stage_idx]

        stage.backward_weight_one_chunk(
            microbatch_index=self.microbatch_idx
        )

    @property
    def work_type(self) -> ActionWorkType:
        return ActionWorkType.compute

    @property
    def has_backward_work(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"{self.stage_idx}W{self.microbatch_idx}"


@dataclasses.dataclass(frozen=True, slots=True)
class ComposeAction(ActionBase):
    """
    Composite action scheduling multiple sub-actions sequentially.

    Used for forward/backward overlapping.

    Attributes:
        actions: A tuple of sub-actions to be executed sequentially.
    """

    actions: tuple[ActionBase, ...]

    def apply(self, ctx: ActionContext):
        for act in self.actions:
            act.apply(ctx)

    @property
    def work_type(self) -> ActionWorkType:
        sub_work_types = {x.work_type for x in self.actions}
        if len(sub_work_types) != 1:
            raise ValueError("")
        return next(iter(sub_work_types))

    @property
    def has_backward_work(self) -> bool:
        return any(x.has_backward_work for x in self.actions)

    def __str__(self) -> str:
        return "|".join(map(str, self.actions))
