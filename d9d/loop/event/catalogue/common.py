import dataclasses
from typing import TYPE_CHECKING

from torch import nn

from d9d.core.dist_context import DistributedContext
from d9d.core.protocol import MicrobatchPackStream

if TYPE_CHECKING:
    from d9d.loop.component import JobSchedule


@dataclasses.dataclass(kw_only=True)
class EventStepContext:
    """Context providing step information during iterative execution.

    Attributes:
        schedule: Object responsible for tracking current step and total steps.
    """

    schedule: "JobSchedule"


@dataclasses.dataclass(kw_only=True)
class EventConfigurationStartedContext:
    """Context provided when the loop configuration process originates.

    Attributes:
        dist_context: The initialized distributed execution context.
    """

    dist_context: DistributedContext


@dataclasses.dataclass(kw_only=True)
class EventDataStreamReadyContext:
    """Context provided when the microbatch pack stream has been fully initialized.

    Attributes:
        stream: The microbatch pack stream instance.
    """

    stream: MicrobatchPackStream


@dataclasses.dataclass(kw_only=True)
class EventModelStagesReadyContext:
    """Context provided when the model stages are initialized and parallelized.

    Attributes:
        modules: The references to the model stages.
    """

    modules: list[nn.Module]
