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
    ForwardComputeAction,
)


class ZeroBubbleVPipelineProgramBuilder(PipelineProgramBuilder):
    """Builder for the Zero Bubble V (ZBV) pipeline schedule.

    This schedule hosts exactly two stages per rank in a V shape. It splits the backward pass into
    input-gradient and weight-gradient parts to fill pipeline bubbles.

    References:
        [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241), Section 6
    """

    def __init__(self):
        """Constructs the ``ZeroBubbleVPipelineProgramBuilder`` object."""

    def compose(self, num_microbatches: int, pp_size: int) -> dict[int, list[ActionBase]]:
        num_stages = self.num_stages_per_rank * pp_size

        # V style: rank 0 hosts stages 0 and N-1, rank 1 hosts stages 1 and N-2, and so on.
        stage_to_rank = build_stage_to_host_rank_topology(pp_size=pp_size, num_stages=num_stages, style=ScheduleStyle.v)

        actions: dict[int, list[ActionBase]] = {}

        for rank in range(pp_size):
            actions[rank] = self._generate_rank_schedule(
                rank=rank,
                pp_size=pp_size,
                num_stages=num_stages,
                target_microbatches=num_microbatches,
            )

        return add_communication_ops(compute_actions=actions, stage_to_rank=stage_to_rank, num_stages=num_stages)

    def _generate_rank_schedule(  # noqa: C901 - the schedule phases share counters
        self,
        rank: int,
        pp_size: int,
        num_stages: int,
        target_microbatches: int,
    ) -> list[ActionBase]:
        # The phase bounds assume a full pipeline. Simulate enough microbatches to fill it, then drop
        # the extra microbatches at the end.
        simulated_n_micro = max(2 * pp_size - 1, target_microbatches)

        rank_ops: list[ActionBase] = []

        # -- Stage Identification (V-Shape) --
        # s0: the chunk on the way down the V (stage 0 on rank 0).
        # s1: the chunk on the way back up the V (stage N-1 on rank 0).
        s0 = rank
        s1 = num_stages - 1 - rank

        # -- Counters --
        # Next microbatch index per chunk. f: forward, b: input backward, w: weight backward.
        f0_cnt = 0
        b0_cnt = 0
        w0_cnt = 0

        f1_cnt = 0
        b1_cnt = 0
        w1_cnt = 0

        # -- Helpers --

        def emit_f(stage: int, idx: int):
            rank_ops.append(ForwardComputeAction(stage_idx=stage, microbatch_idx=idx))

        def emit_i_and_w(stage: int, idx: int):
            rank_ops.append(BackwardFullInputComputeAction(stage_idx=stage, microbatch_idx=idx, full_backward=False))
            rank_ops.append(BackwardWeightComputeAction(stage_idx=stage, microbatch_idx=idx))

        def emit_i(stage: int, idx: int):
            rank_ops.append(BackwardFullInputComputeAction(stage_idx=stage, microbatch_idx=idx, full_backward=False))

        def emit_w(stage: int, idx: int):
            rank_ops.append(BackwardWeightComputeAction(stage_idx=stage, microbatch_idx=idx))

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
            if f0_cnt < simulated_n_micro:
                emit_f(s0, f0_cnt)
                f0_cnt += 1

            emit_i_and_w(s0, b0_cnt)
            b0_cnt += 1
            w0_cnt += 1

            emit_f(s1, f1_cnt)
            f1_cnt += 1

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
            emit_i(s0, b0_cnt)
            b0_cnt += 1

            # A weight backward deferred by an earlier input backward.
            emit_w(s0, w0_cnt)
            w0_cnt += 1

        # -- Phase 7: Flush Remaining Weights --
        while w1_cnt < b1_cnt:
            emit_w(s1, w1_cnt)
            w1_cnt += 1

        while w0_cnt < b0_cnt:
            emit_w(s0, w0_cnt)
            w0_cnt += 1

        # -- Integrity Check --
        if not (w0_cnt == b0_cnt == f0_cnt):
            raise RuntimeError(
                f"The ZBV schedule for chunk 0 is inconsistent: the forward ({f0_cnt}), input backward ({b0_cnt}) "
                f"and weight backward ({w0_cnt}) counts must be equal."
            )
        if not (w1_cnt == b1_cnt == f1_cnt):
            raise RuntimeError(
                f"The ZBV schedule for chunk 1 is inconsistent: the forward ({f1_cnt}), input backward ({b1_cnt}) "
                f"and weight backward ({w1_cnt}) counts must be equal."
            )

        # -- Post-Process: Filter to Target Microbatches --
        # Drop the actions on the extra simulated microbatches.
        final_ops: list[ActionBase] = []
        for action in rank_ops:
            if isinstance(action, (ForwardComputeAction, BackwardFullInputComputeAction, BackwardWeightComputeAction)):
                if action.microbatch_idx < target_microbatches:
                    final_ops.append(action)
            else:
                final_ops.append(action)

        return final_ops

    @property
    def num_stages_per_rank(self) -> int:
        return 2

    @property
    def topology_style(self) -> ScheduleStyle:
        return ScheduleStyle.v
