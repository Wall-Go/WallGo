import pytest
import numpy as np
import WallGo


def f_analytic(x):
    # a simple function to test derivatives
    return x * np.sin(x)


def dfdx_analytic(x):
    # the first derivative, analytically
    return np.sin(x) + x * np.cos(x)


def d2fdx2_analytic(x):
    # the second derivative, analytically
    return 2 * np.cos(x) - x * np.sin(x)


@pytest.fixture
def xRange():
    # the values of x where to test the derivative
    return np.linspace(-10, 10, num=100)


@pytest.mark.parametrize(
    "n, order, bounded, aTol",
    [
        (1, 2, False, 1e-6),
        (1, 2, True, 1e-3),
        (2, 2, False, 1e-6),
        (2, 2, True, 1e-2),
        (1, 4, False, 1e-12),
        (1, 4, True, 1e-12),
        (2, 4, False, 1e-12),
        (2, 4, True, 1e-12),
    ]
)
def test_derivative(
    xRange, n: int, order: int, bounded: bool, aTol: float,
):
    """
    Tests accuracy of derivative function
    """

    # bounds?
    if bounded:
        bounds = (min(xRange), max(xRange))
    else:
        bounds = None

    # expected result
    if n == 1:
        deriv_analytic = dfdx_analytic(xRange)
    elif n == 2:
        deriv_analytic = d2fdx2_analytic(xRange)
    else:
        raise WallGo.WallGoError(
            f"derivative function supports n=1,2, not {n=}"
        )

    # testing first derivatives
    deriv_WallGo = WallGo.helpers.derivative(
        f_analytic, xRange, n=n, order=order, bounds=bounds
    )
    np.testing.assert_allclose(deriv_WallGo, deriv_analytic, atol=aTol)
