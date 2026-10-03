import abc
import dataclasses
from typing import Generic, Self, TypeVar

from pydantic import BaseModel
from torch import nn

from d9d.model_state.mapper import ModelStateMapper


@dataclasses.dataclass(slots=True)
class PeftInjectionResult:
    """Encapsulates the result of injecting a PEFT method into a model.

    Attributes:
        parameters_to_train: The parameters that must stay trainable.
        load_state_mappers: The mappers that load pre-trained weights into the modified structure.
    """

    parameters_to_train: list[nn.Parameter]
    load_state_mappers: list[ModelStateMapper]


TConfig = TypeVar("TConfig", bound=BaseModel)


class PeftMethod(abc.ABC, Generic[TConfig]):
    """Base class for all Parameter-Efficient Fine-Tuning methods."""

    @abc.abstractmethod
    def inject(self, module: nn.Module) -> PeftInjectionResult:
        """Modifies the module in place to apply the PEFT method.

        Args:
            module: The PyTorch module to modify.

        Returns:
            The trainable parameters and the mappers for loading weights into the new structure.
        """
        ...

    @abc.abstractmethod
    def merge(self, module: nn.Module):
        """Merges the trained adapters back into the base model parameters.

        Args:
            module: The PyTorch module to update.
        """
        ...

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: TConfig) -> Self:
        """Creates the method from a configuration object.

        Args:
            config: The configuration object.

        Returns:
            The method instance.
        """
        ...
