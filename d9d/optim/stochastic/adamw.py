from typing import cast

import torch
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT, StateDict

from d9d.kernel.stochastic import adamw_stochastic_bf16_

_GENERATOR_STATE_KEY = "_d9d_generator_state"


def _new_buffer(p: torch.Tensor, dtype_override: torch.dtype) -> torch.Tensor:
    if isinstance(p, DTensor):
        local_p = p.to_local()
    else:
        local_p = p

    out = torch.zeros_like(local_p, dtype=dtype_override).contiguous()

    if isinstance(p, DTensor):
        out = DTensor.from_local(
            local_tensor=out,
            device_mesh=p.device_mesh,
            placements=p.placements,
            run_check=False,
            shape=p.shape,
            stride=p.stride(),
        )

    return out


def _tensor_to_local(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


class StochasticAdamW(Optimizer):
    """Implements the AdamW algorithm with Stochastic Rounding.

    This optimizer is designed to handle stochastic rounding primarily for BF16 training,
    leveraging a custom kernel.

    Parameters must be in BF16. Gradients could be both in BF16 and FP32.

    It natively supports PyTorch distributed ``DTensor`` parameters.

    It maintains its own random number generator state to ensure reproducibility.
    """

    def __init__(
            self,
            params: ParamsT,
            lr: float,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-2,
            generator: torch.Generator | None = None,
            state_dtype: torch.dtype = torch.float32,
    ):
        """Constructs a new StochasticAdamW optimizer.

         Args:
             params: Iterable of parameters to optimize or dicts defining parameter groups.
             lr: Learning rate.
             betas: Coefficients used for computing running averages of gradient and its square.
             eps: Term added to the denominator to improve numerical stability.
             weight_decay: Weight decay coefficient.
             generator: Pseudorandom number generator for stochastic rounding. If None,
                 a new generator is created and seeded from the main PyTorch generator.
             state_dtype: Data Type to use for the optimizer states.
         """

        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if generator is None:
            generator = torch.Generator(device="cpu")
            # make the generator fork from pytorch's main generator
            seed = cast(int, torch.randint(0, 2**32, (1,)).item())
            generator.manual_seed(seed)

        self._generator = generator

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "state_dtype": state_dtype
        }
        super().__init__(params, defaults)

    def state_dict(self) -> StateDict:
        state_dict = super().state_dict()
        state_dict[_GENERATOR_STATE_KEY] = self._generator.get_state()
        return state_dict

    def load_state_dict(self, state_dict: StateDict) -> None:
        if _GENERATOR_STATE_KEY in state_dict:
            self._generator.set_state(state_dict.pop(_GENERATOR_STATE_KEY))
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure: None = None) -> None:  # type: ignore[override]
        if closure is not None:
            raise ValueError("Closure is not supported")

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            state_dtype = group["state_dtype"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("StochasticAdamW does not support sparse gradients")

                state = self.state[p]

                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = _new_buffer(p, dtype_override=state_dtype)
                    state["exp_avg_sq"] = _new_buffer(p, dtype_override=state_dtype)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                adamw_stochastic_bf16_(
                    params=_tensor_to_local(p),
                    grads=_tensor_to_local(grad),
                    exp_avg=_tensor_to_local(exp_avg),
                    exp_avg_sq=_tensor_to_local(exp_avg_sq),
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                    step=state["step"],
                    generator=self._generator
                )
