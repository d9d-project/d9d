from ..component.program import (
    PipelineProgramBuilder,
    ScheduleStyle,
    add_communication_ops,
    build_stage_to_host_rank_topology,
)
from ..component.runtime import (
    ActionBase,
    BackwardFullInputComputeAction,
    ForwardComputeAction,
)


class LoopedBFSPipelineProgramBuilder(PipelineProgramBuilder):
    """
    Builder for the Breadth-First Pipeline Parallelism schedule.

    This schedule runs all available forward microbatches for local stages first.
    If configured for training, it then runs backwards in reverse topological order.

    References:
        https://arxiv.org/pdf/2211.05953
    """

    def __init__(self, num_stages_per_rank: int, inference_mode: bool = False):
        """
        Constructs the LoopedBFS builder.

        Args:
            num_stages_per_rank: Number of stages per rank.
            inference_mode: If True, only forward passes are scheduled. If False,
                both forward and backward passes are scheduled.
        """
        self._num_stages_per_rank = num_stages_per_rank
        self._inference_mode = inference_mode

    def compose(self, num_microbatches: int, pp_size: int) -> dict[int, list[ActionBase]]:
        num_stages = self._num_stages_per_rank * pp_size
        stage_to_rank = build_stage_to_host_rank_topology(
            pp_size=pp_size,
            num_stages=num_stages,
            style=ScheduleStyle.loop
        )

        compute_actions: dict[int, list[ActionBase]] = {r: [] for r in range(pp_size)}

        for rank in range(pp_size):
            my_stages = [s for s in range(num_stages) if stage_to_rank[s] == rank]

            # Schedule all Forwards
            # In Breadth-First loops, we finish all microbatches for the current stage
            # before moving to the next stage assigned to this rank.
            for stage_idx in my_stages:
                for mb_idx in range(num_microbatches):
                    compute_actions[rank].append(
                        ForwardComputeAction(
                            stage_idx=stage_idx,
                            microbatch_idx=mb_idx
                        )
                    )

            # Schedule all Backwards (Reverse order) - Only if training
            if not self._inference_mode:
                for stage_idx in reversed(my_stages):
                    for mb_idx in reversed(range(num_microbatches)):
                        compute_actions[rank].append(
                            BackwardFullInputComputeAction(
                                stage_idx=stage_idx,
                                microbatch_idx=mb_idx,
                                full_backward=True
                            )
                        )

        return add_communication_ops(
            compute_actions=compute_actions,
            stage_to_rank=stage_to_rank,
            num_stages=num_stages
        )

    @property
    def num_stages_per_rank(self) -> int:
        return self._num_stages_per_rank

    @property
    def topology_style(self) -> ScheduleStyle:
        return ScheduleStyle.loop
