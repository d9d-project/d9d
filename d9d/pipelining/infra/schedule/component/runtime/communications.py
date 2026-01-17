import torch.distributed as dist

from d9d.pipelining.infra.stage import PipelineStage


def _schedule_batched_p2p(ops: list[dist.P2POp]) -> list[dist.Work]:
    if not len(ops):
        return []

    return dist.batch_isend_irecv(ops)


def _wait_batched_p2p(work: list[dist.Work]):
    for work_item in work:
        work_item.wait()


class PipelineCommunicationHandler:
    """Manages point-to-point communications between pipeline stages."""

    def __init__(self, stages: dict[int, PipelineStage]):
        """
        Constructs the communication handler.

        Args:
            stages: Mapping of stage indices to PipelineStage instances.
        """

        self._stages = stages

        self._forward_receive_ops: dict[tuple[int, int], list[dist.Work]] = {}
        self._backward_receive_ops: dict[tuple[int, int], list[dist.Work]] = {}

        self._send_ops: list[list[dist.Work]] = []

    def schedule_fwd_send(self, stage_idx: int, microbatch_idx: int):
        """Schedules non-blocking connection to send forward pass outputs."""

        stage = self._stages[stage_idx]
        work = _schedule_batched_p2p(stage.get_fwd_send_ops(microbatch_idx))
        self._send_ops.append(work)

    def schedule_bwd_send(self, stage_idx: int, microbatch_idx: int):
        """Schedules non-blocking connection to send backward pass outputs."""

        stage = self._stages[stage_idx]
        work = _schedule_batched_p2p(stage.get_bwd_send_ops(microbatch_idx))
        self._send_ops.append(work)

    def schedule_fwd_recv(self, stage_idx: int, microbatch_idx: int):
        """
        Schedules non-blocking connection to receive forward pass inputs.

        Raises:
            ValueError: If a receive op is already pending for this stage/microbatch.
        """
        stage = self._stages[stage_idx]
        key = (stage_idx, microbatch_idx)

        if key in self._forward_receive_ops:
            raise ValueError()

        work = _schedule_batched_p2p(stage.get_fwd_recv_ops(microbatch_idx))
        self._forward_receive_ops[key] = work

    def wait_fwd_recv(self, stage_idx: int, microbatch_idx: int):
        """Blocks until the forward pass receive operation completes."""
        key = (stage_idx, microbatch_idx)
        _wait_batched_p2p(self._forward_receive_ops.pop(key))

    def schedule_bwd_recv(self, stage_idx: int, microbatch_idx: int):
        """
        Schedules non-blocking connection to receive backward pass inputs.

        Raises:
            ValueError: If a receive op is already pending for this stage/microbatch.
        """

        stage = self._stages[stage_idx]
        key = (stage_idx, microbatch_idx)

        if key in self._backward_receive_ops:
            raise ValueError()

        work = _schedule_batched_p2p(stage.get_bwd_recv_ops(microbatch_idx))

        self._backward_receive_ops[key] = work

    def wait_bwd_recv(self, stage_idx: int, microbatch_idx: int):
        """Blocks until the backward pass receive operation completes."""

        key = (stage_idx, microbatch_idx)
        _wait_batched_p2p(self._backward_receive_ops.pop(key))

    def wait_send_all(self):
        """Blocks until all pending send operations are completed."""

        while self._send_ops:
            ops = self._send_ops.pop()
            for op in ops:
                op.wait()
