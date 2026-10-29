import math
from typing import Annotated, Literal

import torch
import torch.nn.functional as F
from fla.modules.conv import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda.gate import fused_kda_gate
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from d9d.kernel.swiglu import silu_mul
from d9d.module.base import ModuleLateInit
from d9d.module.block.normalization import RMSNorm


class CausalShortDepthwiseConv1d(nn.Module, ModuleLateInit):
    """Causal 1D depthwise convolution (short convolution) as used in Mamba and FLA architectures.

    Applies a depthwise 1D convolution with left padding, so each position sees only past positions,
    followed by a SiLU activation.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
    ) -> None:
        """Constructs the ``CausalShortDepthwiseConv1d`` object.

        Args:
            hidden_size: Number of input and output channels.
            kernel_size: Size of the convolution kernel.
        """
        super().__init__()
        self._kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, kernel_size))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Applies the causal short convolution.

        Args:
            x: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            mask: Optional padding mask. Masked positions are zeroed before the convolution.
                Shape: ``(batch, seq_len)``.

        Returns:
            Output tensor. Shape: ``(batch, seq_len, hidden_size)``.
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        x, _ = causal_conv1d(
            x=x,
            weight=self.weight,
            bias=None,
            output_final_state=False,
            activation="silu",
            backend="triton",
        )  # ty: ignore[call-non-callable] - fla-core has wrong type annotations for causal_conv1d

        return x

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class LogSigmoidDecayGate(nn.Module, ModuleLateInit):
    """Decay gate that uses a scaled log-sigmoid.

    Used in GLA, the original DeltaNet and HGRN-2.
    """

    def __init__(self, hidden_size: int, num_heads: int, normalizer: float = 16.0) -> None:
        """Constructs the ``LogSigmoidDecayGate`` object.

        Args:
            hidden_size: Hidden size.
            num_heads: Number of attention heads (output dimension).
            normalizer: Temperature that divides the log-sigmoid output.
        """
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_heads, bias=False)
        self._normalizer = normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the decay gate.

        Args:
            x: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.

        Returns:
            Decay gate in log space, with values in ``(-inf, 0]``. Shape: ``(batch, seq_len, num_heads)``.
        """
        return F.logsigmoid(self.proj(x)) / self._normalizer

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        self.proj.reset_parameters()


class MambaDecayGate(nn.Module, ModuleLateInit):
    """Mamba-style decay gate with learnable ``A_log`` and ``dt_bias``.

    Used in Mamba, Mamba-2, Qwen3-Next and Qwen3.5.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        normalizer: float = 16.0,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> None:
        """Constructs the ``MambaDecayGate`` object.

        Args:
            hidden_size: Hidden size.
            num_heads: Number of attention heads.
            normalizer: Upper bound of the uniform initialization of ``A``.
            dt_min: Minimum dt for initialization.
            dt_max: Maximum dt for initialization.
            dt_init_floor: Floor for dt clamping during initialization.
        """
        super().__init__()

        self.proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))

        self._num_heads = num_heads
        self._normalizer = normalizer
        self._dt_min = dt_min
        self._dt_max = dt_max
        self._dt_init_floor = dt_init_floor

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the decay gate.

        Args:
            x: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.

        Returns:
            Decay gate in log space, with values in ``(-inf, 0]``. Shape: ``(batch, seq_len, num_heads)``.
        """
        gk = self.proj(x).unsqueeze(-1)
        gate = fused_kda_gate(gk, A_log=self.A_log, dt_bias=self.dt_bias)

        return gate.squeeze(-1)

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        self.proj.reset_parameters()

        nn.init.uniform_(self.A_log, 0.0, self._normalizer)
        self.A_log.data = torch.log(self.A_log.data)

        dt = torch.exp(
            torch.rand(self._num_heads, device=self.dt_bias.device) * (math.log(self._dt_max) - math.log(self._dt_min))
            + math.log(self._dt_min)
        ).clamp(min=self._dt_init_floor)
        self.dt_bias.data = dt + torch.log(-torch.expm1(-dt))


class MambaDecayGateParameters(BaseModel):
    """Configuration for the Mamba-style decay gate.

    Attributes:
        type: Discriminator field. Always ``"mamba"``.
        normalizer: Upper bound of the uniform initialization of ``A``.
        dt_min: Minimum ``dt`` for initialization.
        dt_max: Maximum ``dt`` for initialization.
        dt_init_floor: Floor for ``dt`` clamping during initialization.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["mamba"] = "mamba"
    normalizer: float
    dt_min: float
    dt_max: float
    dt_init_floor: float


class LogSigmoidDecayGateParameters(BaseModel):
    """Configuration for the log-sigmoid decay gate.

    Attributes:
        type: Discriminator field. Always ``"logsigmoid"``.
        normalizer: Temperature that divides the log-sigmoid output.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["logsigmoid"] = "logsigmoid"
    normalizer: float


AnyDecayGateParameters = Annotated[
    MambaDecayGateParameters | LogSigmoidDecayGateParameters,
    Field(discriminator="type"),
]


def _build_decay_gate(
    config: AnyDecayGateParameters,
    hidden_size: int,
    num_heads: int,
) -> MambaDecayGate | LogSigmoidDecayGate:
    """Builds a decay gate module for the given configuration.

    Args:
        config: Decay gate configuration.
        hidden_size: Hidden size.
        num_heads: Number of attention heads.

    Returns:
        The decay gate module.

    Raises:
        ValueError: If the decay gate configuration type is unknown.
    """
    match config:
        case MambaDecayGateParameters():
            return MambaDecayGate(
                hidden_size=hidden_size,
                num_heads=num_heads,
                normalizer=config.normalizer,
                dt_min=config.dt_min,
                dt_max=config.dt_max,
                dt_init_floor=config.dt_init_floor,
            )
        case LogSigmoidDecayGateParameters():
            return LogSigmoidDecayGate(
                hidden_size=hidden_size,
                num_heads=num_heads,
                normalizer=config.normalizer,
            )
        case _:
            raise ValueError(f"Unknown decay gate config type ({type(config)}).")


class GatedDeltaNet(nn.Module, ModuleLateInit):
    """Gated DeltaNet (GDN) attention layer.

    The layer combines linear attention based on the delta rule with Mamba-style data-dependent gating and
    short causal convolutions. It runs these steps:

    1.  Linear projections for Q, K, V, the output gate, the decay gate and the write strength (beta).
    2.  Causal short depthwise convolution on Q, K and V.
    3.  Data-dependent decay (Mamba-style or log-sigmoid).
    4.  GQA/MQA head expansion for Q and K.
    5.  Chunked gated delta rule, with optional L2 normalization of Q and K.
    6.  Per-head RMSNorm and SiLU-gated output projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_query_key_heads: int,
        num_value_heads: int,
        head_qk_dim: int,
        head_v_dim: int,
        norm_eps: float,
        conv_size: int,
        decay_gate: AnyDecayGateParameters,
        use_qk_l2norm: bool = True,
    ) -> None:
        """Constructs the ``GatedDeltaNet`` object.

        Args:
            hidden_size: Hidden size.
            num_query_key_heads: Number of query and key heads before grouped expansion.
            num_value_heads: Number of value heads.
            head_qk_dim: Dimension of a single query or key head.
            head_v_dim: Dimension of a single value head.
            norm_eps: Epsilon for the output RMSNorm.
            conv_size: Kernel size of the short causal convolution.
            decay_gate: Decay gate configuration.
            use_qk_l2norm: Whether to L2-normalize Q and K inside the kernel.

        Raises:
            ValueError: If ``num_value_heads`` is not divisible by ``num_query_key_heads``.
        """
        super().__init__()

        if num_value_heads % num_query_key_heads != 0:
            raise ValueError(
                f"num_value_heads ({num_value_heads}) must be divisible by num_query_key_heads ({num_query_key_heads})."
            )

        self._hidden_size = hidden_size
        self._num_qk_heads = num_query_key_heads
        self._num_v_heads = num_value_heads
        self._num_qk_groups = num_value_heads // num_query_key_heads
        self._head_qk_dim = head_qk_dim
        self._head_v_dim = head_v_dim
        self._use_qk_l2norm = use_qk_l2norm

        q_dim = num_query_key_heads * head_qk_dim
        k_dim = num_query_key_heads * head_qk_dim
        v_dim = num_value_heads * head_v_dim

        self._qkv_split_sizes = [q_dim, k_dim, v_dim]

        self.qkv_proj = nn.Linear(hidden_size, q_dim + k_dim + v_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, v_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_value_heads, bias=False)
        self.decay_gate = _build_decay_gate(
            config=decay_gate,
            hidden_size=hidden_size,
            num_heads=num_value_heads,
        )

        self.qkv_conv1d = CausalShortDepthwiseConv1d(q_dim + k_dim + v_dim, conv_size)

        self.out_norm = RMSNorm(head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(v_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes Gated DeltaNet attention.

        Args:
            hidden_states: Input tensor. Shape: ``(batch, seq_len, hidden_size)``.
            attention_mask: Optional padding mask. Shape: ``(batch, seq_len)``.

        Returns:
            Output tensor. Shape: ``(batch, seq_len, hidden_size)``.
        """
        b, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        qkv = self.qkv_conv1d(self.qkv_proj(hidden_states))

        q, k, v = torch.split(qkv, self._qkv_split_sizes, dim=-1)

        gk = self.decay_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        q = q.view(b, seq_len, self._num_qk_heads, self._head_qk_dim)
        k = k.view(b, seq_len, self._num_qk_heads, self._head_qk_dim)
        v = v.view(b, seq_len, self._num_v_heads, self._head_v_dim)

        if self._num_qk_groups > 1:
            q = (
                q.unsqueeze(3)
                .expand(-1, -1, -1, self._num_qk_groups, -1)
                .reshape(b, seq_len, self._num_v_heads, self._head_qk_dim)
            )
            k = (
                k.unsqueeze(3)
                .expand(-1, -1, -1, self._num_qk_groups, -1)
                .reshape(b, seq_len, self._num_v_heads, self._head_qk_dim)
            )

        out, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=gk,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=self._use_qk_l2norm,
        )

        out = self.out_norm(out)
        out = out.reshape(b, seq_len, -1)
        out = silu_mul(self.g_proj(hidden_states), out)

        return self.o_proj(out)

    def reset_parameters(self) -> None:
        """Resets module parameters."""
        self.qkv_proj.reset_parameters()
        self.g_proj.reset_parameters()
        self.b_proj.reset_parameters()
        self.decay_gate.reset_parameters()
        self.o_proj.reset_parameters()
        self.qkv_conv1d.reset_parameters()
        self.out_norm.reset_parameters()
