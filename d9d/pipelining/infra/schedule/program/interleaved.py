from collections import defaultdict, deque

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


class Interleaved1F1BPipelineProgramBuilder(PipelineProgramBuilder):
    """Builder for interleaved pipeline parallelism schedules.

    This builder supports:

    1.  **Standard Interleaved 1F1B**: Hosts several stages per rank and prefers depth-first
        execution. (See https://arxiv.org/abs/2104.04473)
    2.  **Interleaved Zero Bubble (ZB1P)**: Extends 1F1B by splitting the backward pass into
        input-gradient and weight-gradient parts. Weight gradients are deferred to fill pipeline
        bubbles. (See https://arxiv.org/abs/2401.10241)
    """

    def __init__(self, num_stages_per_rank: int, enable_zero_bubble: bool = False):
        """Constructs the ``Interleaved1F1BPipelineProgramBuilder`` object.

        Args:
            num_stages_per_rank: Number of stages per rank.
            enable_zero_bubble: If ``True``, uses the ZB1P variant, which splits backward passes
                to reduce bubbles.
        """
        self._num_stages_per_rank = num_stages_per_rank
        self._enable_zero_bubble = enable_zero_bubble

    def _get_warmup_ops(
        self,
        rank: int,
        microbatches_per_round: int,
        pp_size: int,
        n_microbatches: int,
        multiply_factor: int,
    ) -> int:
        """Calculates the number of warmup steps required before entering steady state.

        Returns:
            The number of warmup operations.
        """
        warmups_ops_last_stage = (self._num_stages_per_rank - 1) * microbatches_per_round
        warmup_ops = warmups_ops_last_stage + multiply_factor * ((pp_size - 1) - rank)
        return min(warmup_ops, n_microbatches * self._num_stages_per_rank)

    def compose(self, num_microbatches: int, pp_size: int) -> dict[int, list[ActionBase]]:
        """Generates the execution program for all ranks.

        Args:
            num_microbatches: Number of microbatches per step. It must be divisible by the number of
                rounds, ``max(1, num_microbatches // pp_size)``.
            pp_size: Number of pipeline ranks.

        Returns:
            A dictionary mapping rank indices to their list of sequential actions.

        Raises:
            ValueError: If ``num_microbatches`` is not divisible by the number of rounds.
        """
        num_stages = self.num_stages_per_rank * pp_size

        if num_stages % pp_size != 0:
            raise ValueError(
                f"num_stages ({num_stages}) must be divisible by pp_size ({pp_size}) for interleaved schedules."
            )

        # 1. Topology Setup
        # Round-robin: rank 0 hosts stages 0, pp_size, 2 * pp_size, and so on.
        stage_to_rank = build_stage_to_host_rank_topology(
            pp_size=pp_size, num_stages=num_stages, style=ScheduleStyle.loop
        )

        num_rounds = max(1, num_microbatches // pp_size)

        if num_microbatches % num_rounds != 0:
            raise ValueError(
                f"num_microbatches ({num_microbatches}) must be divisible by the number of rounds ({num_rounds})."
            )

        microbatches_per_round = num_microbatches // num_rounds

        # 2. Schedule Generation
        actions: dict[int, list[ActionBase]] = {}

        # Zero Bubble 1F1B uses a shorter warmup (factor 1) than standard 1F1B (factor 2).
        warmup_multiplier = 1 if self._enable_zero_bubble else 2

        for rank in range(pp_size):
            actions[rank] = self._generate_rank_schedule(
                rank=rank,
                pp_size=pp_size,
                n_microbatches=num_microbatches,
                microbatches_per_round=microbatches_per_round,
                multiply_factor=warmup_multiplier,
            )

        # 3. Communication Injection
        return add_communication_ops(
            compute_actions=actions,
            stage_to_rank=stage_to_rank,
            num_stages=num_stages,
        )

    def _generate_rank_schedule(  # noqa: C901 - the phases share counters and the weight queue
        self,
        rank: int,
        pp_size: int,
        n_microbatches: int,
        microbatches_per_round: int,
        multiply_factor: int,
    ) -> list[ActionBase]:
        """Generates the sequential list of compute actions for a specific rank.

        Returns:
            A list of compute actions.
        """
        rank_actions: list[ActionBase] = []

        # -- State Tracking --
        # Map: stage_idx -> next_microbatch_idx
        fwd_counters: dict[int, int] = defaultdict(int)
        bwd_counters: dict[int, int] = defaultdict(int)

        # FIFO queue for deferred weight gradients in Zero Bubble.
        # Stores: (stage_idx, microbatch_idx).
        pending_weights: deque[tuple[int, int]] = deque()

        # -- Helpers --

        def get_global_stage(local_idx: int) -> int:
            """Converts a local stage index on this rank to a global stage index.

            Returns:
                The global stage index.
            """
            return (local_idx * pp_size) + rank

        def get_fwd_local_idx(op_idx: int) -> int:
            return (op_idx // microbatches_per_round) % self._num_stages_per_rank

        def get_bwd_local_idx(op_idx: int, warmup_offset: int) -> int:
            return (
                self._num_stages_per_rank
                - 1
                - ((op_idx - warmup_offset) // microbatches_per_round) % self._num_stages_per_rank
            )

        def emit_forward(op_idx: int):
            local_idx = get_fwd_local_idx(op_idx)
            stage = get_global_stage(local_idx)
            mb = fwd_counters[stage]

            rank_actions.append(ForwardComputeAction(stage_idx=stage, microbatch_idx=mb))
            fwd_counters[stage] += 1

        def emit_backward(op_idx: int, warmup_offset: int):
            local_idx = get_bwd_local_idx(op_idx, warmup_offset)
            stage = get_global_stage(local_idx)
            mb = bwd_counters[stage]

            # Zero Bubble runs the input backward now and defers the weight backward.
            # Standard 1F1B runs the full backward now.
            is_full = not self._enable_zero_bubble

            rank_actions.append(
                BackwardFullInputComputeAction(stage_idx=stage, microbatch_idx=mb, full_backward=is_full)
            )

            if self._enable_zero_bubble:
                pending_weights.append((stage, mb))

            bwd_counters[stage] += 1

        def try_emit_weight_zb(op_idx: int, warmup_offset: int):
            if not self._enable_zero_bubble or not pending_weights:
                return

            steps_into_1f1b = op_idx - warmup_offset
            # Deeper ranks wait longer before they start the deferred weight backwards.
            if steps_into_1f1b >= rank:
                w_stage, w_mb = pending_weights.popleft()
                rank_actions.append(BackwardWeightComputeAction(stage_idx=w_stage, microbatch_idx=w_mb))

        # -- Execution Phase Math --

        warmup_ops = self._get_warmup_ops(rank, microbatches_per_round, pp_size, n_microbatches, multiply_factor)
        total_microbatch_ops = self._num_stages_per_rank * n_microbatches
        fwd_bwd_ops = total_microbatch_ops - warmup_ops
        cooldown_ops = total_microbatch_ops - fwd_bwd_ops

        # Combine into one sequence for iteration, but handle logic per phase.
        total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops

        # -- Main Schedule Loop --

        for op in range(total_ops):
            # Phase 1: Warmup (Forward Only)
            if op < warmup_ops:
                emit_forward(op)

            # Phase 2: Steady State (1F1B)
            elif op < warmup_ops + fwd_bwd_ops:
                emit_forward(op)
                emit_backward(op, warmup_offset=warmup_ops)
                try_emit_weight_zb(op, warmup_offset=warmup_ops)

            # Phase 3: Cooldown (Backward Only)
            else:
                emit_backward(op, warmup_offset=warmup_ops)
                try_emit_weight_zb(op, warmup_offset=warmup_ops)

        # -- Post-Loop: Flush Remaining Weights (ZB only) --
        while pending_weights:
            w_stage, w_mb = pending_weights.popleft()
            rank_actions.append(BackwardWeightComputeAction(stage_idx=w_stage, microbatch_idx=w_mb))

        return rank_actions

    @property
    def num_stages_per_rank(self) -> int:
        return self._num_stages_per_rank

    @property
    def topology_style(self) -> ScheduleStyle:
        return ScheduleStyle.loop
