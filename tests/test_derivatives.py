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

def fMultivariate_analytic(x, scaleRatio=1):
    # a simple multivariate function to test gradient and hessian
    return np.sin(x[...,0])*np.cos(x[...,1]/scaleRatio)

def gradient_fMultivariate_analytic(x, scaleRatio=1):
    # the gradient, analytically
    grad = np.zeros_like(x)
    grad[...,0] = np.cos(x[...,0])*np.cos(x[...,1]/scaleRatio)
    grad[...,1] = -np.sin(x[...,0])*np.sin(x[...,1]/scaleRatio)/scaleRatio
    return grad

def hessian_fMultivariate_analytic(x, scaleRatio=1):
    # the hessian, analytically
    hess = np.zeros(x.shape+x.shape[-1:])
    hess[...,0,0] = -np.sin(x[...,0])*np.cos(x[...,1]/scaleRatio)
    hess[...,0,1] = -np.cos(x[...,0])*np.sin(x[...,1]/scaleRatio)/scaleRatio
    hess[...,1,0] = -np.cos(x[...,0])*np.sin(x[...,1]/scaleRatio)/scaleRatio
    hess[...,1,1] = -np.sin(x[...,0])*np.cos(x[...,1]/scaleRatio)/scaleRatio**2
    return hess


@pytest.fixture
def xRange():
    # the values of x where to test the derivative
    return np.linspace(-10, 10, num=100)

@pytest.fixture
def multivariateRange():
    xRange = np.linspace(-10, 10, 100)
    yRange = np.linspace(-10, 10, 100)
    X,Y = np.meshgrid(xRange, yRange)
    result = np.zeros(X.shape+(2,))
    result[...,0] = X
    result[...,1] = Y
    return result


@pytest.mark.parametrize(
    "n, order, bounded, rTol",
    [
        (1, 2, False, 1e-6),
        (1, 2, True, 1e-3),
        (2, 2, False, 1e-6),
        (2, 2, True, 1e-2),
        (1, 4, False, 1e-10),
        (1, 4, True, 1e-7),
        (2, 4, False, 1e-8),
        (2, 4, True, 1e-7),
    ]
)
def test_derivative(
    xRange, n: int, order: int, bounded: bool, rTol: float,
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
    np.testing.assert_allclose(deriv_WallGo, deriv_analytic, atol=0, rtol=rTol)
    
@pytest.mark.parametrize(
    "order, scaleRatio, rTol, axis",
    [
         (2, 1, 1e-8,None),
         (4, 1, 1e-10,None),
         (2, 100, 1e-8,None),
         (4, 100, 1e-10,None),
         (2, 1, 1e-8,[0]),
         (4, 1, 1e-10,[0]),
         (2, 100, 1e-8,[0]),
         (4, 100, 1e-10,[0]),
         (2, 1, 1e-8,[1]),
         (4, 1, 1e-10,[1]),
         (2, 100, 1e-8,[1]),
         (4, 100, 1e-10,[1]),
     ]
)
def test_gradient(multivariateRange, order: int, scaleRatio: float, rTol: float, axis: list):
    """
    Tests accuracy of gradient function
    """
    gradient_analytic = gradient_fMultivariate_analytic(multivariateRange, scaleRatio)
    if axis is not None:
        gradient_analytic = gradient_analytic[...,axis]
    gradient_WallGo = WallGo.helpers.gradient(fMultivariate_analytic, multivariateRange, order, 1e-12, [1,scaleRatio], axis=axis, args=(scaleRatio,))
    np.testing.assert_allclose(gradient_analytic, gradient_WallGo, atol=0, rtol=rTol)
    
@pytest.mark.parametrize(
    "order, scaleRatio, rTol, axis",
    [
         (2, 1, 1e-6, None),
         (4, 1, 1e-7, None),
         (2, 100, 1e-6, None),
         (4, 100, 1e-7, None),
         (2, 1, 1e-6, [0]),
         (4, 1, 1e-7, [0]),
         (2, 100, 1e-6, [0]),
         (4, 100, 1e-7, [0]),
         (2, 1, 1e-6, [1]),
         (4, 1, 1e-7, [1]),
         (2, 100, 1e-6, [1]),
         (4, 100, 1e-7, [1]),
     ]
)
def test_hessian(multivariateRange, order: int, scaleRatio: float, rTol: float, axis: int):
    """
    Tests accuracy of hessian function
    """
    hessian_analytic = hessian_fMultivariate_analytic(multivariateRange, scaleRatio)
    if axis is not None:
        hessian_analytic = hessian_analytic[...,axis]
    hessian_WallGo = WallGo.helpers.hessian(fMultivariate_analytic, multivariateRange, order, 1e-12, [1,scaleRatio], yAxis=axis, args=(scaleRatio,))
    np.testing.assert_allclose(hessian_analytic, hessian_WallGo, atol=0, rtol=rTol)
