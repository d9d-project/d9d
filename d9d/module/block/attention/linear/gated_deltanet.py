import math
from typing import Annotated, Literal

import torch
import torch.nn.functional as F
from fla.modules.conv import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda.gate import fused_kda_gate
from pydantic import BaseModel, Field
from torch import nn

from d9d.kernel.swiglu import silu_mul
from d9d.module.base import ModuleLateInit
from d9d.module.block.normalization import RMSNorm


class CausalShortDepthwiseConv1d(nn.Module, ModuleLateInit):
    """
    Causal 1D depthwise convolution (short convolution) as used in Mamba/FLA architectures.
    Applies a grouped (depthwise) 1D convolution with left-padding to ensure causality,
    followed by an optional activation function.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
    ) -> None:
        """Constructs a CausalShortDepthwiseConv1d object.

        Args:
            hidden_size: Number of input and output channels.
            kernel_size: Size of the convolution kernel.
        """
        super().__init__()
        self._kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, kernel_size))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Runs the forward pass for causal short depthwise convolution.

        Args:
            x: Input tensor of shape `(batch, seq_len, hidden_size)`.
            mask: Optional attention mask of shape `(batch, seq_len)`.

        Returns:
            Output tensor of shape `(batch, seq_len, hidden_size)`.
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
        )  # ty:ignore[call-non-callable]  -- fla-core has bad typings unfortunately

        return x

    def reset_parameters(self) -> None:
        """Resets the learnable parameters."""

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


class LogSigmoidDecayGate(nn.Module, ModuleLateInit):
    """
    Decay gate using scaled log-sigmoid.
    Used in GLA, original Delta Net, HGRN-2.
    """

    def __init__(self, hidden_size: int, num_heads: int, normalizer: float = 16.0) -> None:
        """Constructs a LogSigmoidDecayGate object.

        Args:
            hidden_size: Input dimension.
            num_heads: Number of attention heads (output dimension).
            normalizer: Temperature τ dividing the logsigmoid output.
        """
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_heads, bias=False)
        self._normalizer = normalizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass for LogSigmoidDecayGate.

        Args:
            x: Input tensor of shape `(batch, seq_len, hidden_size)`.

        Returns:
            Decay gate in log-space of shape `(batch, seq_len, num_heads)`, with values
                in `(-∞, 0]`.
        """
        return F.logsigmoid(self.proj(x)) / self._normalizer

    def reset_parameters(self) -> None:
        """Resets the learnable parameters."""

        self.proj.reset_parameters()


class MambaDecayGate(nn.Module, ModuleLateInit):
    """
    Mamba-style decay gate with learnable A_log and dt_bias.
    Used in Mamba, Mamba-2, Qwen3-Next, Qwen3.5.
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
        """Constructs a MambaDecayGate object.

        Args:
            hidden_size: Input dimension.
            num_heads: Number of attention heads.
            normalizer: Upper bound for uniform A initialization.
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
        """Runs the forward pass for MambaDecayGate.

        Args:
            x: Input tensor of shape `(batch, seq_len, hidden_size)`.

        Returns:
            Decay gate in log-space of shape `(batch, seq_len, num_heads)`, with values
                in `(-∞, 0]`.
        """

        gk = self.proj(x).unsqueeze(-1)
        gate = fused_kda_gate(gk, A_log=self.A_log, dt_bias=self.dt_bias)

        return gate.squeeze(-1)

    def reset_parameters(self) -> None:
        """Resets the learnable parameters for this module."""

        self.proj.reset_parameters()

        nn.init.uniform_(self.A_log, 0.0, self._normalizer)
        self.A_log.data = torch.log(self.A_log.data)

        dt = torch.exp(
            torch.rand(self._num_heads, device=self.dt_bias.device) * (math.log(self._dt_max) - math.log(self._dt_min))
            + math.log(self._dt_min)
        ).clamp(min=self._dt_init_floor)
        self.dt_bias.data = dt + torch.log(-torch.expm1(-dt))


class MambaDecayGateParameters(BaseModel):
    """Configuration parameters for the Mamba-style decay gate."""

    type: Literal["mamba"] = "mamba"
    normalizer: float
    dt_min: float
    dt_max: float
    dt_init_floor: float


class LogSigmoidDecayGateParameters(BaseModel):
    """Configuration parameters for the LogSigmoid decay gate."""

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
    """Constructs a decay gate module based on the provided configuration.

    Args:
        config: Decay gate configuration object determining gate type and settings.
        hidden_size: Hidden size.
        num_heads: Number of attention heads.

    Returns:
        An instantiated decay gate module.

    Raises:
        ValueError: If an unknown decay gate configuration type is provided.
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
            raise ValueError(f"Unknown decay gate config type: {type(config)}")


class GatedDeltaNet(nn.Module, ModuleLateInit):
    """
    Implements Gated DeltaNet (GDN) attention mechanism.

    This module combines linear attention based on the Delta Rule with Mamba-style
    data-dependent gating and short causal convolutions.

    Pipeline:
        1.  Linear projections for Q, K, V, output gate (G), decay gate (GK), and
            write strength (Beta).
        2.  Causal short depthwise convolution applied to Q, K, V.
        3.  Data-dependent decay computation (Mamba-style or log-sigmoid).
        4.  GQA/MQA head expansion for K and V.
        5.  Chunked Gated Delta Rule (with optional internal L2 norm on Q/K).
        6.  Per-head RMSNorm and SiLU-gated output projection.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_qk_dim: int,
        head_v_dim: int,
        norm_eps: float,
        conv_size: int,
        decay_gate: AnyDecayGateParameters,
        use_qk_l2norm: bool = True,
    ) -> None:
        """Constructs a GatedDeltaNet object.

        Args:
            hidden_size: Hidden size.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key and value attention heads.
            head_qk_dim: Dimension allocated for a single query or key per head.
            head_v_dim: Dimension allocated for a single value per head.
            norm_eps: Small constant added for numerical stability to the normalization layer.
            conv_size: Size of the causal convolution kernel context.
            decay_gate: Structured parameters to initialize the selected decay gate mechanism.
            use_qk_l2norm: Whether to enable L2 normalization applied to Q/K internally.

        Raises:
            ValueError: When num_attention_heads is not uniformly divisible by num_key_value_heads.
        """
        super().__init__()

        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads})."
            )

        self._hidden_size = hidden_size
        self._num_heads = num_attention_heads
        self._num_kv_heads = num_key_value_heads
        self._num_kv_groups = num_attention_heads // num_key_value_heads
        self._head_qk_dim = head_qk_dim
        self._head_v_dim = head_v_dim
        self._use_qk_l2norm = use_qk_l2norm

        qk_dim = num_attention_heads * head_qk_dim
        kv_qk_dim = num_key_value_heads * head_qk_dim
        kv_v_dim = num_key_value_heads * head_v_dim
        v_dim = num_attention_heads * head_v_dim

        self._qkv_split_sizes = [qk_dim, kv_qk_dim, kv_v_dim]

        # --- Linear projections ---
        self.qkv_proj = nn.Linear(hidden_size, qk_dim + kv_qk_dim + kv_v_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, v_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_attention_heads, bias=False)
        self.decay_gate = _build_decay_gate(
            config=decay_gate,
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
        )

        # --- Short causal convolutions ---
        self.qkv_conv1d = CausalShortDepthwiseConv1d(qk_dim + kv_qk_dim + kv_v_dim, conv_size)

        # --- Output normalization & projection ---
        self.out_norm = RMSNorm(head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(v_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Runs forward pass.

        Args:
            hidden_states: Input tensor sequence of shape `(batch, seq_len, hidden_size)`.
            attention_mask: Optional padding mask tensor of shape `(batch, seq_len)`.

        Returns:
            Processed tensor possessing the identical shape as the input.
        """
        b, seq_len, _ = hidden_states.shape

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        qkv = self.qkv_conv1d(self.qkv_proj(hidden_states))

        q, k, v = torch.split(qkv, self._qkv_split_sizes, dim=-1)

        gk = self.decay_gate(hidden_states)
        beta = torch.sigmoid(self.b_proj(hidden_states))

        q = q.view(b, seq_len, self._num_heads, self._head_qk_dim)
        k = k.view(b, seq_len, self._num_kv_heads, self._head_qk_dim)
        v = v.view(b, seq_len, self._num_kv_heads, self._head_v_dim)

        if self._num_kv_groups > 1:
            k = (
                k.unsqueeze(3)
                .expand(-1, -1, -1, self._num_kv_groups, -1)
                .reshape(b, seq_len, self._num_heads, self._head_qk_dim)
            )
            v = (
                v.unsqueeze(3)
                .expand(-1, -1, -1, self._num_kv_groups, -1)
                .reshape(b, seq_len, self._num_heads, self._head_v_dim)
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
        """Resets learnable parameters of this module."""
        self.qkv_proj.reset_parameters()
        self.g_proj.reset_parameters()
        self.b_proj.reset_parameters()
        self.decay_gate.reset_parameters()
        self.o_proj.reset_parameters()
        self.qkv_conv1d.reset_parameters()
        self.out_norm.reset_parameters()
