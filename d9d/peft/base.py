import abc
import dataclasses
from typing import Generic, Self, TypeVar

from pydantic import BaseModel
from torch import nn

from d9d.model_state.mapper import ModelStateMapper


@dataclasses.dataclass(slots=True)
class PeftInjectionResult:
    """
    Encapsulates the result of injecting a PEFT method into a model.

    Attributes:
        parameters_to_train: A list of parameters that should remain trainable.
        load_state_mappers: A list of mappers required to load pre-trained weights into the modified structure.
    """

    parameters_to_train: list[nn.Parameter]
    load_state_mappers: list[ModelStateMapper]


TConfig = TypeVar("TConfig", bound=BaseModel)


class PeftMethod(abc.ABC, Generic[TConfig]):
    """
    Abstract base class for all Parameter-Efficient Fine-Tuning methods.
    """

    @abc.abstractmethod
    def inject(self, module: nn.Module) -> PeftInjectionResult:
        """
        Modifies the module in-place to apply the PEFT strategy.

        Args:
            module: The PyTorch module to modify.

        Returns:
            Result object containing trainable parameters and structure mappers.
        """
        ...

    @abc.abstractmethod
    def merge(self, module: nn.Module):
        """
        Merges the trained adapters back into the base model parameters.

        Args:
            module: The PyTorch module to update.
        """

        ...

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: TConfig) -> Self:
        """
        Creates an instance of the method from a configuration object.

        Args:
            config: The configuration object.

        Returns:
            An instance of the PeftMethod.
        """

        ...
