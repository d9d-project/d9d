from enum import StrEnum


class SleepTag(StrEnum):
    """Subsystems that ``Trainer.sleep`` and ``Trainer.wake`` act on.

    Attributes:
        TENSOR_STATES: All GPU tensor state: model parameters and buffers, optimizer state, gradient buckets and
            the residual loss accumulator. They are always offloaded together.
        COMMS: NCCL process groups. Opt-in. It is not implemented yet, so requesting it raises
            ``NotImplementedError``.
    """

    TENSOR_STATES = "tensor_states"
    COMMS = "comms"


DEFAULT_SLEEP_TAGS = frozenset({SleepTag.TENSOR_STATES})
"""The default tags for ``Trainer.sleep`` and ``Trainer.wake``: tensor state only, no comms."""
