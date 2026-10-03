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
    """AdamW optimizer for bf16 parameters with stochastic rounding.

    Parameters must be bf16. Gradients can be bf16 or fp32. Parameters can be ``DTensor``s.

    The optimizer owns its random number generator and saves the generator state in ``state_dict``. A
    resumed run therefore gets the same rounding noise.
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
        """Constructs the ``StochasticAdamW`` object.

        Args:
            params: Parameters to optimize, or dicts that define parameter groups.
            lr: Learning rate.
            betas: Decay rates of the first and second moments.
            eps: Term added to the denominator for numerical stability.
            weight_decay: Decoupled weight decay coefficient.
            generator: Random number generator for stochastic rounding. If ``None``, a new CPU generator is
                seeded from the default PyTorch generator.
            state_dtype: Dtype of the optimizer states. Can be ``torch.float32`` or ``torch.bfloat16``.

        Raises:
            ValueError: If ``lr``, ``eps``, ``betas`` or ``weight_decay`` is out of range.
        """
        if lr <= 0:
            raise ValueError(f"lr ({lr}) must be positive.")
        if eps <= 0:
            raise ValueError(f"eps ({eps}) must be positive.")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"betas[0] ({betas[0]}) must be in [0.0, 1.0).")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas[1] ({betas[1]}) must be in [0.0, 1.0).")
        if weight_decay < 0:
            raise ValueError(f"weight_decay ({weight_decay}) must be non-negative.")

        if generator is None:
            generator = torch.Generator(device="cpu")
            # Seed from the default generator, so torch.manual_seed also fixes the rounding noise.
            seed = cast(int, torch.randint(0, 2**32, (1,)).item())
            generator.manual_seed(seed)

        self._generator = generator

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "state_dtype": state_dtype,
        }
        super().__init__(params, defaults)

    def state_dict(self) -> StateDict:
        """Returns the optimizer state, including the generator state."""
        state_dict = super().state_dict()
        state_dict[_GENERATOR_STATE_KEY] = self._generator.get_state()
        return state_dict

    def load_state_dict(self, state_dict: StateDict) -> None:
        """Loads the optimizer state, including the generator state if it is present."""
        if _GENERATOR_STATE_KEY in state_dict:
            self._generator.set_state(state_dict.pop(_GENERATOR_STATE_KEY))
        super().load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure: None = None) -> None:  # ty: ignore[invalid-method-override] - closures are not supported
        if closure is not None:
            raise ValueError("StochasticAdamW does not support closures. Call step() without a closure.")

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
                    raise RuntimeError("StochasticAdamW does not support sparse gradients.")

                state = self.state[p]

                # Initialize the state lazily on the first step of each parameter.
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
                    generator=self._generator,
                )
