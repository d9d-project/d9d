from collections import deque
from typing import Deque

from d9d.pipelining.infra.schedule.component import PipelineProgramBuilder, ActionBase, \
    build_stage_to_host_rank_topology, ScheduleStyle, ForwardComputeAction, BackwardFullInputComputeAction, \
    BackwardWeightComputeAction, ComposeAction, add_communication_ops


class DualPipeVPipelineProgramBuilder(PipelineProgramBuilder):
    """
    Builder for the DualPipeV Pipeline Parallelism schedule.

    DualPipeV is a specialized bi-directional pipeline schedule designed for high
    throughput training. It requires exactly 2 stages per pipeline rank (V-shape)
    and utilizes split backward passes (Input gradients vs Weight gradients)
    to fill pipeline bubbles.

    References:
        https://github.com/deepseek-ai/DualPipe
        https://hackmd.io/@ufotalent/r1lVXsa9Jg
    """

    def __init__(self):
        """
        Constructs the DualPipeV builder.
        """
        pass

    def compose(
            self, num_stages: int, num_microbatches: int, pp_size: int
    ) -> dict[int, list[ActionBase]]:
        if num_stages != pp_size * 2:
            raise ValueError(
                f"DualPipeV requires exactly 2 stages per rank (total {pp_size * 2}), "
                f"but got {num_stages} stages for {pp_size} ranks."
            )
        if num_microbatches < num_stages:
            raise ValueError(
                f"DualPipeV requires num_microbatches ({num_microbatches}) >= "
                f"num_stages ({num_stages})."
            )

        # Ranks hold stages in a V pattern (e.g., Rank 0 holds Stage 0 and Stage N-1).
        # We rely on the sorted order of local steps to determine Phase 0 (Forward-going)
        # and Phase 1 (Backward-coming).
        stage_to_rank = build_stage_to_host_rank_topology(
            pp_size=pp_size, num_stages=num_stages, style=ScheduleStyle.v
        )

        compute_actions: dict[int, list[ActionBase]] = {r: [] for r in range(pp_size)}

        for rank in range(pp_size):
            # Identify local stages: s0 is Phase 0, s1 is Phase 1
            my_stages = sorted([s for s, r in stage_to_rank.items() if r == rank])
            s0, s1 = my_stages[0], my_stages[1]

            # Track microbatch indices for each stage and operation type
            # f_idx: Next Forward microbatch
            # b_idx: Next Backward microbatch (Input or Full)
            f_idx = {s0: 0, s1: 0}
            b_idx = {s0: 0, s1: 0}

            # Queue for Zero Bubble optimization: stores (stage, mb_idx) for deferred weight grads
            weight_queue: Deque[tuple[int, int]] = deque()

            # --- Helper Functions for Action Emission ---

            def _add_f(stage: int):
                compute_actions[rank].append(
                    ForwardComputeAction(stage_idx=stage, microbatch_idx=f_idx[stage])
                )
                f_idx[stage] += 1

            def _add_b_full(stage: int):
                compute_actions[rank].append(
                    BackwardFullInputComputeAction(
                        stage_idx=stage,
                        microbatch_idx=b_idx[stage],
                        full_backward=True,
                    )
                )
                b_idx[stage] += 1

            def _add_b_input(stage: int):
                mb = b_idx[stage]
                compute_actions[rank].append(
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
                compute_actions[rank].append(
                    BackwardWeightComputeAction(stage_idx=s, microbatch_idx=mb)
                )

            def _add_overlap_f_b(stage_f: int, stage_b: int, b_is_full: bool):
                """Emit overlapped Forward and Backward actions."""
                mb_f = f_idx[stage_f]
                mb_b = b_idx[stage_b]

                act_f = ForwardComputeAction(stage_idx=stage_f, microbatch_idx=mb_f)

                act_b = BackwardFullInputComputeAction(
                    stage_idx=stage_b, microbatch_idx=mb_b, full_backward=b_is_full
                )
                if not b_is_full:
                    weight_queue.append((stage_b, mb_b))

                f_idx[stage_f] += 1
                b_idx[stage_b] += 1

                # Note: d9d infra treats ComposeAction sequentially in simulation,
                # but runtime may overlap them.
                compute_actions[rank].append(ComposeAction(actions=(act_f, act_b)))

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
                _add_b_input(s1)  # Backward Input Phase 1
                _pop_w()  # Weight Phase (accumulated from prev)
                _add_f(s1)  # Forward Phase 1

            # Step 4: The Main Loop (Interleaved Forward/Backward)
            step_4 = num_microbatches - 2 * pp_size + rank + 1
            for i in range(step_4):
                # Sub-step A: F0 & B1
                if i == 0 and rank == pp_size - 1:
                    # Specific case for last rank on first iter: do not overlap
                    _add_f(s0)
                    _add_b_full(s1)
                else:
                    # Overlap F0 and B1 (usually full backward unless we were in ZB mode,
                    # but DualPipeV main loop defaults to full for simplicity unless tuned)
                    # DeepSeek impl uses standard backward here (zb=False).
                    _add_overlap_f_b(stage_f=s0, stage_b=s1, b_is_full=True)

                # Sub-step B: F1 & B0
                # Overlap F1 and B0 (Full)
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
                # DeepSeek source explicitly uses enable_zb=True here for chunk 0
                _add_b_input(s0)

            # Step 8: Flush Weights
            step_8 = rank + 1
            for _ in range(step_8):
                _pop_w()

        # 4. Inject Communication Operations
        # This wrapper handles dependency analysis and inserts Send/Recv/Wait ops.
        return add_communication_ops(
            compute_actions=compute_actions,
            stage_to_rank=stage_to_rank,
            num_stages=num_stages
        )
