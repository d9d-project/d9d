"""Optimizer provider built from a configuration."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field
from torch.optim import SGD, Adam, AdamW, Optimizer

from d9d.optim.stochastic import StochasticAdamW

from ..config import dtype_from_name
from ..control import InitializeOptimizerStageContext, OptimizerProvider


class StochasticAdamWOptimizerConfig(BaseModel):
    """Configuration for the Stochastic AdamW optimizer.

    Attributes:
        name: Discriminator field. Always ``"stochastic_adamw"``.
        lr: Learning rate.
        betas: Coefficients used for computing running averages of gradient and its square.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay coefficient.
        state_dtype: The dtype of the optimizer states.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["stochastic_adamw"] = "stochastic_adamw"

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    state_dtype: Literal["float32", "bfloat16"]


class AdamWOptimizerConfig(BaseModel):
    """Configuration for the PyTorch AdamW optimizer.

    Attributes:
        name: Discriminator field. Always ``"adamw"``.
        lr: The learning rate.
        betas: Coefficients for computing running averages of gradient and its square.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay coefficient.
        amsgrad: Whether to use the AMSGrad variant.
        maximize: Whether to maximize the objective instead of minimizing it.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["adamw"] = "adamw"

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    maximize: bool = False


class AdamOptimizerConfig(BaseModel):
    """Configuration for the PyTorch Adam optimizer.

    Attributes:
        name: Discriminator field. Always ``"adam"``.
        lr: The learning rate.
        betas: Coefficients for computing running averages of gradient and its square.
        eps: Term added to the denominator to improve numerical stability.
        weight_decay: Weight decay coefficient.
        decoupled_weight_decay: Whether to apply decoupled weight decay.
        amsgrad: Whether to use the AMSGrad variant.
        maximize: Whether to maximize the objective instead of minimizing it.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["adam"] = "adam"

    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    decoupled_weight_decay: bool = False
    amsgrad: bool = False
    maximize: bool = False


class SGDOptimizerConfig(BaseModel):
    """Configuration for the PyTorch SGD optimizer.

    Attributes:
        name: Discriminator field. Always ``"sgd"``.
        lr: The learning rate.
        momentum: Momentum factor.
        dampening: Dampening for momentum.
        weight_decay: Weight decay (L2 penalty).
        nesterov: Enables Nesterov momentum.
        maximize: Whether to maximize the objective instead of minimizing it.
    """

    model_config = ConfigDict(extra="forbid")

    name: Literal["sgd"] = "sgd"

    lr: float
    momentum: float = 0
    dampening: float = 0
    weight_decay: float = 0
    nesterov: bool = False
    maximize: bool = False


AutoOptimizerConfig = Annotated[
    StochasticAdamWOptimizerConfig | AdamWOptimizerConfig | AdamOptimizerConfig | SGDOptimizerConfig,
    Field(discriminator="name"),
]
"""Discriminated union of the supported optimizer configurations."""


class AutoOptimizerProvider(OptimizerProvider):
    """Provider that builds a PyTorch optimizer from a configuration object.

    The PyTorch optimizers (AdamW, Adam and SGD) are built with ``fused=True``.
    """

    def __init__(self, config: AutoOptimizerConfig):
        """Constructs the ``AutoOptimizerProvider`` object."""
        self._config = config

    def __call__(self, context: InitializeOptimizerStageContext) -> Optimizer:
        params = context.model.parameters()
        match self._config:
            case StochasticAdamWOptimizerConfig():
                return StochasticAdamW(
                    params=params,
                    lr=self._config.lr,
                    betas=self._config.betas,
                    eps=self._config.eps,
                    weight_decay=self._config.weight_decay,
                    state_dtype=dtype_from_name(self._config.state_dtype),
                )
            case AdamWOptimizerConfig():
                return AdamW(
                    params=params,
                    lr=self._config.lr,
                    betas=self._config.betas,
                    eps=self._config.eps,
                    weight_decay=self._config.weight_decay,
                    amsgrad=self._config.amsgrad,
                    maximize=self._config.maximize,
                    fused=True,
                )
            case AdamOptimizerConfig():
                return Adam(
                    params=params,
                    lr=self._config.lr,
                    betas=self._config.betas,
                    eps=self._config.eps,
                    weight_decay=self._config.weight_decay,
                    decoupled_weight_decay=self._config.decoupled_weight_decay,
                    amsgrad=self._config.amsgrad,
                    maximize=self._config.maximize,
                    fused=True,
                )
            case SGDOptimizerConfig():
                return SGD(
                    params,
                    lr=self._config.lr,
                    momentum=self._config.momentum,
                    dampening=self._config.dampening,
                    weight_decay=self._config.weight_decay,
                    nesterov=self._config.nesterov,
                    maximize=self._config.maximize,
                    fused=True,
                )
            case _:
                raise ValueError(f"Optimizer configuration type ({type(self._config).__name__}) is not supported.")
