import abc
import dataclasses
from enum import StrEnum
from typing import Generic

from d9d.pipelining.api import TPipelineInput, TPipelineOutput, TSharedInput, TStageTransfer
from d9d.pipelining.infra.stage import PipelineStage

from .callback import PipelineLossHandler, PipelineResultHandler
from .communications import PipelineCommunicationHandler


@dataclasses.dataclass(kw_only=True, slots=True)
class ActionContext(Generic[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
    """Runtime context required to execute a pipeline action.

    Attributes:
        pipeline_inputs_microbatches: Per-microbatch ``PipelineInput``, indexed by microbatch.
        pipeline_shared_microbatches: Per-microbatch ``SharedInput``, indexed by microbatch.
        stages: A mapping of stage indices to the ``PipelineStage`` objects on this rank.
        communications: The handler for P2P communications.
        callback: The handler for either loss computation or result processing.
    """

    pipeline_inputs_microbatches: tuple[TPipelineInput, ...]
    pipeline_shared_microbatches: tuple[TSharedInput, ...]

    stages: dict[int, PipelineStage[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]]
    communications: PipelineCommunicationHandler
    callback: PipelineLossHandler[TPipelineOutput] | PipelineResultHandler[TPipelineOutput]


class ActionWorkType(StrEnum):
    """Types of work that an action performs.

    Attributes:
        compute: The action computes (a forward or backward pass).
        communicate: The action communicates over the network (send or receive).
    """

    compute = "compute"
    communicate = "communicate"


class ActionBase(abc.ABC):
    """Abstract base class for all pipeline schedule actions.

    An action is an atomic unit of work in a pipeline schedule, such as computing a microbatch or
    sending a tensor.
    """

    @abc.abstractmethod
    def apply(self, ctx: ActionContext[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
        """Executes the action logic using the provided context.

        Args:
            ctx: The runtime context containing stages, data, and communication handlers.
        """
        ...

    @property
    @abc.abstractmethod
    def work_type(self) -> ActionWorkType:
        """The type of work this action performs."""
        ...

    @property
    @abc.abstractmethod
    def has_backward_work(self) -> bool:
        """Whether this action is part of the backward pass."""
        ...

    @abc.abstractmethod
    def __str__(self) -> str:
        """Returns a short string form of the action for logs and visualization."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class ForwardSendAction(ActionBase):
    """Action that starts sending the forward outputs of a microbatch.

    Attributes:
        stage_idx: The index of the sending pipeline stage.
        microbatch_idx: The index of the microbatch.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
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
    """Action that starts sending the input gradients of a microbatch.

    Attributes:
        stage_idx: The index of the sending pipeline stage.
        microbatch_idx: The index of the microbatch.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
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
    """Action that starts receiving the forward inputs of a microbatch.

    Attributes:
        stage_idx: The index of the receiving pipeline stage.
        microbatch_idx: The index of the microbatch.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
        ctx.communications.schedule_fwd_recv(self.stage_idx, self.microbatch_idx)

    @property
    def work_type(self) -> ActionWorkType:
        return ActionWorkType.communicate

    @property
    def has_backward_work(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"{self.stage_idx}RECV_F{self.microbatch_idx}"


@dataclasses.dataclass(frozen=True, slots=True)
class BackwardReceiveAction(ActionBase):
    """Action that starts receiving the output gradients of a microbatch.

    Attributes:
        stage_idx: The index of the receiving pipeline stage.
        microbatch_idx: The index of the microbatch.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
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
    """Action that runs the forward pass of a microbatch.

    Attributes:
        stage_idx: The index of the pipeline stage.
        microbatch_idx: The index of the microbatch.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
        stage = ctx.stages[self.stage_idx]

        if not stage.info.is_current_stage_first and self.stage_idx - 1 not in ctx.stages:
            ctx.communications.wait_fwd_recv(self.stage_idx, self.microbatch_idx)

        stage.forward_one_chunk(
            microbatch_index=self.microbatch_idx,
            pipeline_inputs=ctx.pipeline_inputs_microbatches[self.microbatch_idx],
            pipeline_shared=ctx.pipeline_shared_microbatches[self.microbatch_idx],
        )

        if stage.info.is_current_stage_last:
            ctx.callback.trigger(stage.get_pipeline_output(self.microbatch_idx), self.microbatch_idx)
        elif self.stage_idx + 1 in ctx.stages:
            ctx.stages[self.stage_idx + 1].set_local_fwd_input(
                inputs=stage.get_produced_transfer(self.microbatch_idx), microbatch_index=self.microbatch_idx
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
    """Action that runs the backward pass of a microbatch, for the inputs and optionally the weights.

    Attributes:
        stage_idx: The index of the pipeline stage.
        microbatch_idx: The index of the microbatch.
        full_backward: If ``True``, computes gradients for the inputs and the weights. If ``False``,
            computes input gradients and defers the weight gradients to a
            ``BackwardWeightComputeAction``.
    """

    stage_idx: int
    microbatch_idx: int
    full_backward: bool

    def apply(self, ctx: ActionContext[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
        stage = ctx.stages[self.stage_idx]

        if not stage.info.is_current_stage_last and self.stage_idx + 1 not in ctx.stages:
            ctx.communications.wait_bwd_recv(self.stage_idx, self.microbatch_idx)

        if stage.info.is_current_stage_last and isinstance(ctx.callback, PipelineLossHandler):
            loss = ctx.callback.acquire_loss(self.microbatch_idx)
        else:
            loss = None

        stage.backward_one_chunk(microbatch_index=self.microbatch_idx, full_backward=self.full_backward, loss=loss)

        if not stage.info.is_current_stage_first and self.stage_idx - 1 in ctx.stages:
            ctx.stages[self.stage_idx - 1].set_local_bwd_input(
                microbatch_index=self.microbatch_idx, inputs=stage.pop_local_bwd_output(self.microbatch_idx)
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
    """Action that runs the deferred weight backward of a microbatch.

    Attributes:
        stage_idx: The index of the pipeline stage.
        microbatch_idx: The index of the microbatch.
    """

    stage_idx: int
    microbatch_idx: int

    def apply(self, ctx: ActionContext[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
        stage = ctx.stages[self.stage_idx]

        stage.backward_weight_one_chunk(microbatch_index=self.microbatch_idx)

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
    """Composite action that runs several sub-actions in order.

    Schedules use it to mark a forward pass and a backward pass for overlap.

    Attributes:
        actions: A tuple of sub-actions to be executed sequentially.
    """

    actions: tuple[ActionBase, ...]

    def apply(self, ctx: ActionContext[TPipelineInput, TStageTransfer, TSharedInput, TPipelineOutput]):
        for act in self.actions:
            act.apply(ctx)

    @property
    def work_type(self) -> ActionWorkType:
        sub_work_types = {x.work_type for x in self.actions}
        if len(sub_work_types) != 1:
            raise ValueError(
                "The sub-actions of a ComposeAction must share one work type, "
                f"but they have work types ({sorted(sub_work_types)})."
            )
        return next(iter(sub_work_types))

    @property
    def has_backward_work(self) -> bool:
        return any(x.has_backward_work for x in self.actions)

    def __str__(self) -> str:
        return "|".join(map(str, self.actions))
