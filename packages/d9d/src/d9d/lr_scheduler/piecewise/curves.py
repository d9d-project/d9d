import abc
import math


class CurveBase(abc.ABC):
    """Abstract base class for interpolation curves used in scheduling."""

    @abc.abstractmethod
    def compute(self, start: float, end: float, step_p: float) -> float:
        """Computes the interpolated value.

        Args:
            start: Value at the start of the phase.
            end: Value at the end of the phase.
            step_p: Progress through the phase, from 0.0 to 1.0.

        Returns:
            The interpolated value.
        """


class CurveLinear(CurveBase):
    """Linearly interpolates between start and end values."""

    def compute(self, start: float, end: float, step_p: float) -> float:
        return start + (end - start) * step_p


class CurveCosine(CurveBase):
    """Cosine annealing curve (half-period cosine)."""

    def compute(self, start: float, end: float, step_p: float) -> float:
        cos_out = (1 + math.cos(math.pi * step_p)) / 2
        return end + (start - end) * cos_out


class CurvePoly(CurveBase):
    """Polynomial curve along ``step_p ** power``."""

    def __init__(self, power: float):
        """Constructs the ``CurvePoly`` object.

        Args:
            power: Exponent of the polynomial. 1.0 is linear, 2.0 is quadratic.
        """
        self._power = power

    def compute(self, start: float, end: float, step_p: float) -> float:
        p_transformed = step_p**self._power
        return start + (end - start) * p_transformed


class CurveExponential(CurveBase):
    """Exponential curve between the start and end values (linear in log space).

    Start and end values below ``1e-8`` are treated as ``1e-8``.
    """

    def compute(self, start: float, end: float, step_p: float) -> float:
        eps = 1e-8
        safe_start = max(start, eps)
        safe_end = max(end, eps)

        out_log = math.log(safe_start) + (math.log(safe_end) - math.log(safe_start)) * step_p
        return math.exp(out_log)
