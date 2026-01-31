import math

import pytest
from d9d.lr_scheduler.piecewise.curves import CurveCosine, CurveExponential, CurveLinear, CurvePoly


@pytest.mark.local
@pytest.mark.parametrize(
    ("start", "end", "p", "expected"),
    [
        (0.0, 10.0, 0.0, 0.0),
        (0.0, 10.0, 0.5, 5.0),
        (0.0, 10.0, 1.0, 10.0),
        (10.0, 0.0, 0.25, 7.5),
        (5.0, 5.0, 0.3, 5.0),
    ],
)
def test_curve_linear(start, end, p, expected):
    curve = CurveLinear()
    assert math.isclose(curve.compute(start, end, p), expected, rel_tol=1e-5)


@pytest.mark.local
@pytest.mark.parametrize(
    ("start", "end", "p", "expected"),
    [
        (0.0, 10.0, 0.0, 0.0),      # cos(0) = 1 -> ((1+1)/2)=1 -> end + (start-end)*1 = start
        (0.0, 10.0, 1.0, 10.0),     # cos(pi) = -1 -> 0 -> end
        (0.0, 10.0, 0.5, 5.0),      # cos(pi/2) = 0 -> 0.5 -> average
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0, 0.0),
    ]
)
def test_curve_cosine(start, end, p, expected):
    curve = CurveCosine()
    result = curve.compute(start, end, p)
    assert math.isclose(result, expected, rel_tol=1e-5)


@pytest.mark.local
@pytest.mark.parametrize(
    ("power", "start", "end", "p", "expected"),
    [
        (1.0, 0.0, 100.0, 0.5, 50.0),       # Linear
        (2.0, 0.0, 100.0, 0.5, 25.0),       # Quadratic: 0.5^2 = 0.25 -> 25
        (0.5, 0.0, 100.0, 0.25, 50.0),      # Sqrt: 0.25^0.5 = 0.5 -> 50
    ]
)
def test_curve_poly(power, start, end, p, expected):
    curve = CurvePoly(power=power)
    assert math.isclose(curve.compute(start, end, p), expected, rel_tol=1e-5)


@pytest.mark.local
@pytest.mark.parametrize(
    ("start", "end", "p", "expected_fn"),
    [
        # Exp(log(1) + 0.5 * (log(100) - log(1))) = Exp(0 + 0.5 * 4.6) approx Exp(2.3) = 10
        (1.0, 100.0, 0.5, lambda s, e: 10.0),
        (1.0, 1.0, 0.5, lambda s, e: 1.0),
    ]
)
def test_curve_exponential(start, end, p, expected_fn):
    curve = CurveExponential()
    expected = expected_fn(start, end)
    assert math.isclose(curve.compute(start, end, p), expected, rel_tol=1e-5)


@pytest.mark.local
def test_curve_exponential_small_values():
    # Test handling of 0 or near-zero values
    curve = CurveExponential()
    # Should not crash on 0 due to clamp
    res = curve.compute(0.0, 10.0, 0.0)
    assert res > 0
