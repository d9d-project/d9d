import dataclasses
from typing import TYPE_CHECKING

from torch import nn
from torch.utils.data import DataLoader

from d9d.core.dist_context import DistributedContext

if TYPE_CHECKING:
    from d9d.loop.component import Stepper


@dataclasses.dataclass(kw_only=True)
class EventStepContext:
    """
    Context providing step information during iterative execution.

    Attributes:
        stepper: Object responsible for tracking current step and total steps.
    """

    stepper: "Stepper"


@dataclasses.dataclass(kw_only=True)
class EventConfigurationStartedContext:
    """
    Context provided when the loop configuration process originates.

    Attributes:
        dist_context: The initialized distributed execution context.
    """

    dist_context: DistributedContext


@dataclasses.dataclass(kw_only=True)
class EventDataLoaderReadyContext:
    """
    Context provided when the data loader has been fully initialized.

    Attributes:
        data_loader: The data loader instance.
    """

    data_loader: DataLoader


@dataclasses.dataclass(kw_only=True)
class EventModelStagesReadyContext:
    """
    Context provided when the model stages are initialized and parallelized.

    Attributes:
        modules: The references to the model stages.
    """

    modules: list[nn.Module]
