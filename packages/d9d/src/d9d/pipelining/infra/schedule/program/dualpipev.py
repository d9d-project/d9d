from collections import deque

from ..component.program import (
    PipelineProgramBuilder,
    ScheduleStyle,
    add_communication_ops,
    build_stage_to_host_rank_topology,
)
from ..component.runtime import (
    ActionBase,
    BackwardFullInputComputeAction,
    BackwardWeightComputeAction,
    ComposeAction,
    ForwardComputeAction,
)


class DualPipeVPipelineProgramBuilder(PipelineProgramBuilder):
    """Builder for the DualPipeV pipeline parallelism schedule.

    DualPipeV is a bidirectional pipeline schedule. It hosts exactly 2 stages per rank in a V shape.
    It splits some backward passes into input-gradient and weight-gradient parts to fill pipeline
    bubbles.

    References:
        *   [DualPipe](https://github.com/deepseek-ai/DualPipe)
        *   [DualPipeV write-up](https://hackmd.io/@ufotalent/r1lVXsa9Jg)
    """

    def __init__(self):
        """Constructs the ``DualPipeVPipelineProgramBuilder`` object."""

    @staticmethod
    def _build_for_rank(  # noqa: C901 - the schedule steps share counters and the weight queue
        rank: int, stage_to_rank: dict[int, int], num_microbatches: int, pp_size: int
    ) -> list[ActionBase]:
        compute_actions: list[ActionBase] = []

        # Identify local stages: s0 is Phase 0, s1 is Phase 1.
        my_stages = sorted([s for s, r in stage_to_rank.items() if r == rank])
        s0, s1 = my_stages[0], my_stages[1]

        # Track microbatch indices for each stage and operation type.
        # f_idx: Next Forward microbatch.
        # b_idx: Next Backward microbatch (Input or Full).
        f_idx = {s0: 0, s1: 0}
        b_idx = {s0: 0, s1: 0}

        # Deferred weight backwards as (stage, microbatch) pairs, in input-backward order.
        weight_queue: deque[tuple[int, int]] = deque()

        # --- Helper Functions for Action Emission ---

        def _add_f(stage: int):
            compute_actions.append(ForwardComputeAction(stage_idx=stage, microbatch_idx=f_idx[stage]))
            f_idx[stage] += 1

        def _add_b_full(stage: int):
            compute_actions.append(
                BackwardFullInputComputeAction(
                    stage_idx=stage,
                    microbatch_idx=b_idx[stage],
                    full_backward=True,
                )
            )
            b_idx[stage] += 1

        def _add_b_input(stage: int):
            mb = b_idx[stage]
            compute_actions.append(
                BackwardFullInputComputeAction(
                    stage_idx=stage,
                    microbatch_idx=mb,
                    full_backward=False,
                )
            )
            weight_queue.append((stage, mb))
            b_idx[stage] += 1

        def _pop_w():
            if not weight_queue:
                return
            s, mb = weight_queue.popleft()
            compute_actions.append(BackwardWeightComputeAction(stage_idx=s, microbatch_idx=mb))

        def _add_overlap_f_b(stage_f: int, stage_b: int, b_is_full: bool):
            """Emits a forward and a backward action as one ``ComposeAction``."""
            mb_f = f_idx[stage_f]
            mb_b = b_idx[stage_b]

            act_f = ForwardComputeAction(stage_idx=stage_f, microbatch_idx=mb_f)

            act_b = BackwardFullInputComputeAction(stage_idx=stage_b, microbatch_idx=mb_b, full_backward=b_is_full)
            if not b_is_full:
                weight_queue.append((stage_b, mb_b))

            f_idx[stage_f] += 1
            b_idx[stage_b] += 1

            # ComposeAction marks the pair for forward/backward overlap. The runtime currently runs them in order.
            compute_actions.append(ComposeAction(actions=(act_f, act_b)))

        # Step 1: nF0 (Startup Phase 0)
        step_1 = (pp_size - rank - 1) * 2
        for _ in range(step_1):
            _add_f(s0)

        # Step 2: nF0F1 (Forward fill)
        step_2 = rank + 1
        for _ in range(step_2):
            _add_f(s0)
            _add_f(s1)

        # Step 3: nI1W1F1 (Mixed Phase with Zero Bubble)
        step_3 = pp_size - rank - 1
        for _ in range(step_3):
            _add_b_input(s1)
            # Runs the oldest deferred weight backward.
            _pop_w()
            _add_f(s1)

        # Step 4: The Main Loop (Interleaved Forward/Backward)
        step_4 = num_microbatches - 2 * pp_size + rank + 1
        for i in range(step_4):
            # Sub-step A: F0 & B1
            if i == 0 and rank == pp_size - 1:
                # The last rank does not overlap F0 and B1 on the first iteration.
                _add_f(s0)
                _add_b_full(s1)
            else:
                # Full backward, as in the DeepSeek implementation (zb=False).
                _add_overlap_f_b(stage_f=s0, stage_b=s1, b_is_full=True)

            # Sub-step B: F1 & B0
            _add_overlap_f_b(stage_f=s1, stage_b=s0, b_is_full=True)

        # Step 5: Cooldown F1/B0
        step_5 = pp_size - rank - 1
        for _ in range(step_5):
            _add_b_full(s1)
            _add_overlap_f_b(stage_f=s1, stage_b=s0, b_is_full=True)

        # Step 6: Cooldown B1/B0 with Zero Bubble ramp-up
        step_6 = rank + 1
        enable_zb = False
        for i in range(step_6):
            # Phase 1 Backward
            if i == step_6 // 2 and rank % 2 == 1:
                enable_zb = True

            if enable_zb:
                _add_b_input(s1)
            else:
                _add_b_full(s1)

            # Phase 0 Backward
            if i == step_6 // 2 and rank % 2 == 0:
                enable_zb = True

            if enable_zb:
                _add_b_input(s0)
            else:
                _add_b_full(s0)

        # Step 7: Zero Bubble Weights + B0
        step_7 = pp_size - rank - 1
        for _ in range(step_7):
            _pop_w()
            # The DeepSeek implementation uses enable_zb=True here for chunk 0.
            _add_b_input(s0)

        # Step 8: Flush Weights
        step_8 = rank + 1
        for _ in range(step_8):
            _pop_w()

        return compute_actions

    def compose(self, num_microbatches: int, pp_size: int) -> dict[int, list[ActionBase]]:
        num_stages = self.num_stages_per_rank * pp_size

        if num_microbatches < num_stages:
            raise ValueError(
                f"num_microbatches ({num_microbatches}) must be at least num_stages ({num_stages}) for DualPipeV. "
                "Use more microbatches per step or a smaller pp_size."
            )

        # V pattern: rank 0 holds stages 0 and N-1. _build_for_rank uses the sorted local stages to
        # tell phase 0 (forward-going) from phase 1 (backward-coming).
        stage_to_rank = build_stage_to_host_rank_topology(pp_size=pp_size, num_stages=num_stages, style=ScheduleStyle.v)

        compute_actions: dict[int, list[ActionBase]] = {r: [] for r in range(pp_size)}

        for rank in range(pp_size):
            compute_actions[rank] = self._build_for_rank(
                rank=rank, pp_size=pp_size, num_microbatches=num_microbatches, stage_to_rank=stage_to_rank
            )

        return add_communication_ops(
            compute_actions=compute_actions, stage_to_rank=stage_to_rank, num_stages=num_stages
        )

    @property
    def num_stages_per_rank(self) -> int:
        return 2

    @property
    def topology_style(self) -> ScheduleStyle:
        return ScheduleStyle.v
