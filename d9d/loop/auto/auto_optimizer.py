import abc
from abc import ABC
from collections.abc import Iterable
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, Field
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from d9d.loop.control import InitializeOptimizerStageContext, OptimizerProvider
from d9d.optim.stochastic import StochasticAdamW


class BaseAutoOptimizerConfig(BaseModel, ABC):
    """
    Abstract base class for optimizer configurations.
    """

    @abc.abstractmethod
    def build(self, params: Iterable[nn.Parameter]) -> Optimizer:
        """
        Creates the PyTorch optimizer instance.

        Args:
            params: An iterable of model parameters to optimize.

        Returns:
            The instantiated optimizer.
        """
        ...


class StochasticAdamWOptimizerConfig(BaseAutoOptimizerConfig):
    """
    Configuration for the Stochastic AdamW optimizer.

    Attributes:
        name: Discriminator tag.
        lr: Learning rate.
        betas: Coefficients used for computing running averages of gradient and its square.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay coefficient.
        state_dtype: Data Type to use for the optimizer states.
    """
    name: Literal["stochastic_adamw"] = "stochastic_adamw"

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    state_dtype: str

    def build(self, params: Iterable[nn.Parameter]) -> Optimizer:
        """Builds StochasticAdamW with the configured parameters."""
        return StochasticAdamW(
            params=params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            state_dtype=getattr(torch, self.state_dtype)
        )


class AdamWOptimizerConfig(BaseAutoOptimizerConfig):
    """
    Configuration for the PyTorch AdamW optimizer.

    Attributes:
        name: Discriminator tag.
        lr: The learning rate.
        betas: Coefficients for computing running averages of gradient and its square.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay coefficient.
        amsgrad: Whether to use the AMSGrad variant.
        maximize: Whether to maximize the params based on the objective (as opposed to minimizing).
    """
    name: Literal["adamw"] = "adamw"

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    maximize: bool = False

    def build(self, params: Iterable[nn.Parameter]) -> Optimizer:
        """Builds fused AdamW with the configured parameters."""
        return AdamW(
            params=params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
            maximize=self.maximize,
            fused=True
        )


class AdamOptimizerConfig(BaseAutoOptimizerConfig):
    """
    Configuration for the PyTorch Adam optimizer.

    Attributes:
        name: Discriminator tag.
        lr: The learning rate.
        betas: Coefficients for computing running averages of gradient and its square.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay coefficient.
        decoupled_weight_decay: Whether to apply decoupled weight decay.
        amsgrad: Whether to use the AMSGrad variant.
        maximize: Whether to maximize the params based on the objective.
    """
    name: Literal["adam"] = "adam"

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    decoupled_weight_decay: bool = False
    amsgrad: bool = False
    maximize: bool = False

    def build(self, params: Iterable[nn.Parameter]) -> Optimizer:
        """Builds fused Adam with the configured parameters."""
        return Adam(
            params=params,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
            decoupled_weight_decay=self.decoupled_weight_decay,
            amsgrad=self.amsgrad,
            maximize=self.maximize,
            fused=True
        )


class SGDOptimizerConfig(BaseAutoOptimizerConfig):
    """
    Configuration for the PyTorch SGD optimizer.

    Attributes:
        name: Discriminator tag.
        lr: The learning rate.
        momentum: Momentum factor.
        dampening: Dampening for momentum.
        weight_decay: Weight decay (L2 penalty).
        nesterov: Enables Nesterov momentum.
        maximize: Whether to maximize the params based on the objective.
    """
    name: Literal["sgd"] = "sgd"

    lr: float
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False
    maximize: bool = False

    def build(self, params: Iterable[nn.Parameter]) -> Optimizer:
        """Builds fused SGD with the configured parameters."""
        return SGD(
            params,
            lr=self.lr,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
            maximize=self.maximize,
            fused=True
        )


AutoOptimizerConfig = Annotated[
    StochasticAdamWOptimizerConfig |
    AdamWOptimizerConfig |
    AdamOptimizerConfig |
    SGDOptimizerConfig,
    Field(discriminator="name")
]


class AutoOptimizerProvider(OptimizerProvider):
    """
    OptimizerProvider that builds a PyTorch optimizer based on a configuration object.
    """

    def __init__(self, config: AutoOptimizerConfig):
        """Constructs the provider with the given configuration."""
        self._config = config

    def __call__(self, context: InitializeOptimizerStageContext) -> Optimizer:
        return self._config.build(context.model.parameters())
