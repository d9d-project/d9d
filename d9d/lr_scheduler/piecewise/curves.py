import abc
import math


class CurveBase(abc.ABC):
    """
    Abstract base class for interpolation curves used in scheduling.
    """

    @abc.abstractmethod
    def compute(self, start: float, end: float, step_p: float) -> float:
        """
        Calculates the interpolated value.

        Args:
            start: The value at the beginning of the phase.
            end: The value at the end of the phase.
            step_p: Progress fraction through the phase (0.0 to 1.0).

        Returns:
            The interpolated value.
        """


class CurveLinear(CurveBase):
    """
    Linearly interpolates between start and end values.
    """

    def compute(self, start: float, end: float, step_p: float) -> float:
        return start + (end - start) * step_p


class CurveCosine(CurveBase):
    """
    Interpolates using a cosine annealing schedule (half-period cosine).
    """

    def compute(self, start: float, end: float, step_p: float) -> float:
        cos_out = (1 + math.cos(math.pi * step_p)) / 2
        return end + (start - end) * cos_out


class CurvePoly(CurveBase):
    """
    Interpolates using a polynomial function.
    """

    def __init__(self, power: float):
        """
        Constructs a polynomial curve.

        Args:
            power: The exponent of the polynomial. 1.0 is linear, 2.0 is quadratic, etc.
        """

        self._power = power

    def compute(self, start: float, end: float, step_p: float) -> float:
        p_transformed = step_p ** self._power
        return start + (end - start) * p_transformed


class CurveExponential(CurveBase):
    """
    Interpolates exponentially between start and end values (log-space linear).
    """

    def compute(self, start: float, end: float, step_p: float) -> float:
        eps = 1e-8
        safe_start = max(start, eps)
        safe_end = max(end, eps)

        out_log = math.log(safe_start) + (math.log(safe_end) - math.log(safe_start)) * step_p
        return math.exp(out_log)
