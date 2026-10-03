import math
from abc import ABC, abstractmethod

import torch


def _prepare_rope_inverse_frequencies(rope_base: float, inside_dim: int) -> torch.Tensor:
    return rope_base ** (-torch.arange(0, inside_dim, 2, dtype=torch.float32) / inside_dim)


class RopeScaling(ABC):
    """Abstract base class for rotary position embedding (RoPE) scaling strategies."""

    @abstractmethod
    def inverse_frequencies(self, rope_base: int, head_dim: int) -> torch.Tensor:
        """Computes the RoPE inverse frequencies for this scaling strategy.

        Args:
            rope_base: Base of the geometric progression of RoPE frequencies.
            head_dim: Dimensionality of the attention head.

        Returns:
            The inverse frequencies. Shape: ``(head_dim // 2,)``.
        """

    @property
    def attention_mscale(self) -> float:
        """The attention scale multiplier (mscale) applied to the cosine and sine embeddings."""
        return 1.0


class NoRopeScaling(RopeScaling):
    """Strategy that applies no scaling to rotary position embeddings."""

    def inverse_frequencies(self, rope_base: int, head_dim: int) -> torch.Tensor:
        return _prepare_rope_inverse_frequencies(rope_base, head_dim)


class LinearRopeScaling(RopeScaling):
    """Linear scaling strategy for rotary position embeddings."""

    def __init__(self, factor: float) -> None:
        """Constructs the ``LinearRopeScaling`` object.

        Args:
            factor: Linear scaling factor. The inverse frequencies are divided by it.
        """
        self._factor = factor

    def inverse_frequencies(self, rope_base: int, head_dim: int) -> torch.Tensor:
        return _prepare_rope_inverse_frequencies(rope_base, head_dim) / self._factor


class YarnRopeScaling(RopeScaling):
    """YaRN (Yet another RoPE extensioN) scaling strategy for position embeddings.

    References:
        [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
    """

    def __init__(
        self,
        factor: float,
        beta_fast: float,
        beta_slow: float,
        original_max_position_embeddings: int,
    ) -> None:
        """Constructs the ``YarnRopeScaling`` object.

        Args:
            factor: Context extension factor.
            beta_fast: Fast boundary (upper bound) of the frequency ramp, in rotations.
            beta_slow: Slow boundary (lower bound) of the frequency ramp, in rotations.
            original_max_position_embeddings: Original context length of the base model.

        Raises:
            ValueError: If ``beta_fast`` is less than or equal to ``beta_slow``.
        """
        if beta_fast <= beta_slow:
            raise ValueError(f"beta_fast ({beta_fast}) must exceed beta_slow ({beta_slow}).")

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
        [NTK-Aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
    """

    def __init__(self, factor: float) -> None:
        """Constructs the ``NtkRopeScaling`` object.

        Args:
            factor: Sequence length expansion factor.
        """
        self._factor = factor

    def inverse_frequencies(self, rope_base: int, head_dim: int) -> torch.Tensor:
        new_base = float(rope_base * (self._factor ** (head_dim / (head_dim - 2))))
        return _prepare_rope_inverse_frequencies(new_base, head_dim)
