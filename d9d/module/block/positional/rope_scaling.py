import math
from abc import ABC, abstractmethod

import torch


def _prepare_rope_inverse_frequencies(rope_base: float, inside_dim: int) -> torch.Tensor:
    return rope_base ** (-torch.arange(0, inside_dim, 2, dtype=torch.float32) / inside_dim)


class RopeScaling(ABC):
    """Abstract base class for Rotary Position Embedding (RoPE) scaling strategies."""

    @abstractmethod
    def inverse_frequencies(self, rope_base: int, head_dim: int) -> torch.Tensor:
        """Calculates the inverse frequencies for the given RoPE scaling strategy.

        Args:
            rope_base: The base value used for calculating frequencies.
            head_dim: The dimension of the attention head.

        Returns:
            The computed inverse frequencies tensor.
        """

    @property
    def attention_mscale(self) -> float:
        """Calculates the attention multiplier scale.

        Returns:
            The attention multiplier scale.
        """
        return 1.0


class NoRopeScaling(RopeScaling):
    """Strategy that applies no scaling to Rotary Position Embeddings."""

    def inverse_frequencies(self, rope_base: int, head_dim: int) -> torch.Tensor:
        return _prepare_rope_inverse_frequencies(rope_base, head_dim)


class LinearRopeScaling(RopeScaling):
    """Linear scaling strategy for Rotary Position Embeddings."""

    def __init__(self, factor: float) -> None:
        """Constructs a linear RoPE scaling object.

        Args:
            factor: The linear scaling factor to apply.
        """
        self._factor = factor

    def inverse_frequencies(self, rope_base: int, head_dim: int) -> torch.Tensor:
        return _prepare_rope_inverse_frequencies(rope_base, head_dim) / self._factor


class YarnRopeScaling(RopeScaling):
    """YaRN (Yet another RoPE extensioN) scaling strategy for position embeddings.

    References:
        https://arxiv.org/abs/2309.00071
    """

    def __init__(
        self,
        factor: float,
        beta_fast: float,
        beta_slow: float,
        original_max_position_embeddings: int,
    ) -> None:
        """Constructs a YaRN RoPE scaling object.

        Args:
            factor: The context scaling extension factor.
            beta_fast: The fast boundary (upper bound) frequency multiplier.
            beta_slow: The slow boundary (lower bound) frequency multiplier.
            original_max_position_embeddings: The original context limit of the base model.

        Raises:
            ValueError: If beta_fast is less than or equal to beta_slow.
        """
        if beta_fast <= beta_slow:
            raise ValueError(f"beta_fast ({beta_fast}) must exceed beta_slow ({beta_slow})")

        self._factor = factor
        self._beta_fast = beta_fast
        self._beta_slow = beta_slow
        self._original_max_position_embeddings = original_max_position_embeddings

    def inverse_frequencies(self, rope_base: int, head_dim: int) -> torch.Tensor:
        dim_half = head_dim // 2

        inv_freq = _prepare_rope_inverse_frequencies(rope_base, head_dim)

        low = max(self._correction_dim(self._beta_fast, rope_base, head_dim), 0.0)
        high = min(self._correction_dim(self._beta_slow, rope_base, head_dim), dim_half - 1)

        ramp = torch.clamp(
            (torch.arange(dim_half, dtype=torch.float32) - low) / (high - low),
            0.0,
            1.0,
        )
        return torch.lerp(inv_freq, inv_freq / self._factor, ramp)

    def _correction_dim(self, rotations: float, rope_base: int, head_dim: int) -> float:
        return (
            head_dim
            * math.log(self._original_max_position_embeddings / (rotations * 2 * math.pi))
            / (2 * math.log(rope_base))
        )

    @property
    def attention_mscale(self) -> float:
        if self._factor <= 1.0:
            return 1.0
        return 0.1 * math.log(self._factor) + 1.0


class NtkRopeScaling(RopeScaling):
    """NTK-Aware (Neural Tangent Kernel) scaling strategy for position embeddings.

    References:
        https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
    """

    def __init__(self, factor: float) -> None:
        """Constructs an NTK-Aware RoPE scaling object.

        Args:
            factor: The sequence length expansion factor.
        """
        self._factor = factor

    def inverse_frequencies(self, rope_base: int, head_dim: int) -> torch.Tensor:
        new_base = float(rope_base * (self._factor ** (head_dim / (head_dim - 2))))
        return _prepare_rope_inverse_frequencies(new_base, head_dim)
