from d9d.pipelining.infra.schedule.component import (
    PipelineProgramBuilder,
    ActionBase,
    ForwardComputeAction,
    BackwardFullInputComputeAction,
    BackwardWeightComputeAction,
    ScheduleStyle,
    build_stage_to_host_rank_topology,
    add_communication_ops,
)


class ZeroBubbleVPipelineProgramBuilder(PipelineProgramBuilder):
    """
    Builder for the Zero Bubble V (ZBV) Pipeline Schedule.

    This schedule is designed for V-shape topologies (2 stages per rank) and
    utilizes the Zero Bubble optimizations by splitting backward passes.

    It requires exactly two stages
    per rank organized in a V-shape topology and splits backward passes into
    Input and Weight gradients to optimize pipeline throughput.

    References:
        https://arxiv.org/pdf/2401.10241, Section 6
    """

    def __init__(self):
        """Constructs the ZBV builder."""
        pass

    def compose(
            self, num_stages: int, num_microbatches: int, pp_size: int
    ) -> dict[int, list[ActionBase]]:
        if num_stages != 2 * pp_size:
            raise ValueError(
                f"ZBV schedule requires exactly 2 stages per rank (total {2 * pp_size}), "
                f"but got {num_stages}."
            )

        # 1. Topology
        # V-style: Rank 0 gets Stage 0 & Stage N-1. Rank 1 gets Stage 1 & Stage N-2...
        stage_to_rank = build_stage_to_host_rank_topology(
            pp_size=pp_size, num_stages=num_stages, style=ScheduleStyle.v
        )

        actions: dict[int, list[ActionBase]] = {}

        for rank in range(pp_size):
            actions[rank] = self._generate_rank_schedule(
                rank=rank,
                pp_size=pp_size,
                num_stages=num_stages,
                target_microbatches=num_microbatches,
            )

        # 2. Inject Communications
        return add_communication_ops(
            compute_actions=actions,
            stage_to_rank=stage_to_rank,
            num_stages=num_stages
        )

    def _generate_rank_schedule(
            self,
            rank: int,
            pp_size: int,
            num_stages: int,
            target_microbatches: int,
    ) -> list[ActionBase]:
        # ZBV logic assumes the pipeline is fully saturated to define the loop bounds.
        # We simulate enough steps to cover the topology startup, then filter
        # down to the user's requested microbatches at the end.
        simulated_n_micro = max(2 * pp_size - 1, target_microbatches)

        rank_ops: list[ActionBase] = []

        # -- Stage Identification (V-Shape) --
        # s0: The "Forward-going" chunk (e.g., Stage 0 for Rank 0)
        # s1: The "Backward-coming" chunk (e.g., Stage N-1 for Rank 0)
        s0 = rank
        s1 = num_stages - 1 - rank

        # -- Counters --
        # Track next microbatch index for each operation type on each chunk.
        # F: Forward, I: Backward Input, W: Backward Weight
        f0_cnt = 0
        b0_cnt = 0  # Input Grad Counter (Chunk 0)
        w0_cnt = 0  # Weight Grad Counter (Chunk 0)

        f1_cnt = 0
        b1_cnt = 0  # Input Grad Counter (Chunk 1)
        w1_cnt = 0  # Weight Grad Counter (Chunk 1)

        # -- Helpers --

        def emit_f(stage: int, idx: int):
            rank_ops.append(ForwardComputeAction(stage_idx=stage, microbatch_idx=idx))

        def emit_i_and_w(stage: int, idx: int):
            rank_ops.append(
                BackwardFullInputComputeAction(
                    stage_idx=stage, microbatch_idx=idx, full_backward=False
                )
            )
            rank_ops.append(
                BackwardWeightComputeAction(stage_idx=stage, microbatch_idx=idx)
            )

        def emit_i(stage: int, idx: int):
            rank_ops.append(
                BackwardFullInputComputeAction(
                    stage_idx=stage, microbatch_idx=idx, full_backward=False
                )
            )

        def emit_w(stage: int, idx: int):
            rank_ops.append(
                BackwardWeightComputeAction(stage_idx=stage, microbatch_idx=idx)
            )

        # -- Phase 1: Warmup 1 (Chunk 0 Forwards) --
        warmup_n1 = 2 * (pp_size - rank) - 1
        for _ in range(warmup_n1):
            emit_f(s0, f0_cnt)
            f0_cnt += 1

        # -- Phase 2: Warmup 2 (Interleave F1, F0) --
        warmup_n2 = rank
        for _ in range(warmup_n2):
            emit_f(s1, f1_cnt)
            f1_cnt += 1
            emit_f(s0, f0_cnt)
            f0_cnt += 1

        # -- Phase 3: Warmup 3 (F1, then B1 I+W) --
        warmup_n3 = pp_size - rank
        for _ in range(warmup_n3):
            emit_f(s1, f1_cnt)
            f1_cnt += 1

            emit_i_and_w(s1, b1_cnt)
            b1_cnt += 1
            w1_cnt += 1

        # -- Phase 4: Stable State --
        while f1_cnt < f0_cnt or f0_cnt < simulated_n_micro:
            # Emit F0 if within bounds
            if f0_cnt < simulated_n_micro:
                emit_f(s0, f0_cnt)
                f0_cnt += 1

            # Emit B0 (I+W)
            emit_i_and_w(s0, b0_cnt)
            b0_cnt += 1
            w0_cnt += 1

            # Emit F1
            emit_f(s1, f1_cnt)
            f1_cnt += 1

            # Emit B1 (I+W)
            emit_i_and_w(s1, b1_cnt)
            b1_cnt += 1
            w1_cnt += 1

        # -- Phase 5: Cooldown 1 (Splitting I and W) --
        # In cooldown, the I and W streams diverge to fill bubbles.
        cooldown_n1 = rank
        for _ in range(cooldown_n1):
            emit_i(s0, b0_cnt)
            b0_cnt += 1

            emit_i(s1, b1_cnt)
            b1_cnt += 1

        # -- Phase 6: Cooldown 2 (I0, then W0) --
        cooldown_n2 = pp_size - rank
        for _ in range(cooldown_n2):
            # Input Grad Chunk 0
            emit_i(s0, b0_cnt)
            b0_cnt += 1

            # Weight Grad Chunk 0 (delayed from previous steps)
            emit_w(s0, w0_cnt)
            w0_cnt += 1

        # -- Phase 7: Flush Remaining Weights --

        # Flush W1
        while w1_cnt < b1_cnt:
            emit_w(s1, w1_cnt)
            w1_cnt += 1

        # Flush W0
        while w0_cnt < b0_cnt:
            emit_w(s0, w0_cnt)
            w0_cnt += 1

        # -- Integrity Check --
        if not (w0_cnt == b0_cnt == f0_cnt):
            raise RuntimeError(
                f"ZBV Schedule Failed (Chunk 0): F={f0_cnt}, I={b0_cnt}, W={w0_cnt}"
            )
        if not (w1_cnt == b1_cnt == f1_cnt):
            raise RuntimeError(
                f"ZBV Schedule Failed (Chunk 1): F={f1_cnt}, I={b1_cnt}, W={w1_cnt}"
            )

        # -- Post-Process: Filter to Target Microbatches --
        # Remove any actions involving simulated microbatches beyond the user's request.
        final_ops: list[ActionBase] = []
        for action in rank_ops:
            if isinstance(action, (ForwardComputeAction,
                                   BackwardFullInputComputeAction,
                                   BackwardWeightComputeAction)):
                if action.microbatch_idx < target_microbatches:
                    final_ops.append(action)
            else:
                final_ops.append(action)

        return final_ops
