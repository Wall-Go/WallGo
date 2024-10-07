"""
Helper functions that are used by WallGo.
This includes the derivative functions, a function to compute the step size for solving
detonations and other small functions.
"""

from typing import Callable
import numpy as np
from scipy import integrate, special, optimize


FIRST_DERIV_COEFF = {
    "2": np.array([[-0.5, 0.5], [-1, 1], [-1, 1]], dtype=float),
    "4": np.array(
        [
            [1, -8, 8, -1],
            [-4, -6, 12, -2],
            [-22, 36, -18, 4],
            [-4, 18, -36, 22],
            [2, -12, 6, 4],
        ],
        dtype=float,
    )
    / 12,
}
SECOND_DERIV_COEFF = {
    "2": np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]], dtype=float),
    "4": np.array(
        [
            [-1, 16, -30, 16, -1],
            [11, -20, 6, 4, -1],
            [35, -104, 114, -56, 11],
            [11, -56, 114, -104, 35],
            [-1, 4, 6, -20, 11],
        ],
        dtype=float,
    )
    / 12,
}

FIRST_DERIV_POS = {
    "2": np.array([[-1, 1], [0, 1], [-1, 0]], dtype=float),
    "4": np.array(
        [[-2, -1, 1, 2], [-1, 0, 1, 2], [0, 1, 2, 3], [-3, -2, -1, 0], [-2, -1, 0, 1]],
        dtype=float,
    ),
}
SECOND_DERIV_POS = {
    "2": np.array([[-1, 0, 1], [0, 1, 2], [-2, -1, 0]], dtype=float),
    "4": np.array(
        [
            [-2, -1, 0, 1, 2],
            [-1, 0, 1, 2, 3],
            [0, 1, 2, 3, 4],
            [-4, -3, -2, -1, 0],
            [-3, -2, -1, 0, 1],
        ],
        dtype=float,
    ),
}

HESSIAN_POS = {
    "2": np.array([[1, 1, -1, -1], [1, -1, 1, -1]], dtype=float),
    "4": np.array(
        [[2, 2, 1, 1, -1, -1, -2, -2], [2, -2, 1, -1, 1, -1, 2, -2]], dtype=float
    ),
}
HESSIAN_COEFF = {
    "2": np.array([1, -1, -1, 1], dtype=float) / 4,
    "4": np.array([-1, 1, 16, -16, -16, 16, 1, -1], dtype=float) / 48,
}


def derivative(
    f: Callable[..., np.ndarray],
    x: float | np.ndarray,
    n: int=1,
    order: int=4,
    bounds: tuple[float,float] | None=None,
    epsilon: float=1e-16,
    scale: float=1.0,
    dx: float | None=None,
    args: list | None=None,
) -> np.ndarray:
    r"""Computes numerical derivatives of a callable function. Use the epsilon
    and scale parameters to estimate the optimal value of dx, if the latter is
    not provided.


    Parameters
    ----------
    f : function
        Function to differentiate. Should take a float or an array as argument
        and return a float or array (the returned array can have a different
        shape as the input, but the first axis must match).
    x : float or array-like
        The position at which to evaluate the derivative.
    n : int, optional
        The number of derivatives to take. Can be 0, 1, 2. The default is 1.
    order : int, optional
        The accuracy order of the scheme. Errors are of order
        :math:`\mathcal{O}({\rm d}x^{\text{order}/(\text{order+n})})`. Can be 2 or 4.
        Note that the order at the endpoints is reduced by 1 as it would require
        more function evaluations to keep the same order. The default is 4.
    bounds : tuple or None, optional
        Interval in which f can be called. If None, can be evaluated anywhere.
        The default is None.
    epsilon : float, optional
        Fractional accuracy at which f can be evaluated. If f is a simple
        function, should be close to the machine precision. Default is 1e-16.
    scale : float, optional
        Typical scale at which f(x) change by order 1. Default is 1.
    dx : float or None, optional
        The magnitude of finite differences. If None, use epsilon and scale to
        estimate the optimal dx. Default is None.
    args: list, optional
        List of other fixed arguments passed to the function :math:`f`.

    Returns
    -------
    res : float
        The value of the derivative of :py:data:`f` evaluated at :py:data:`x`.

    """
    x = np.asarray(x)

    boundsTuple: tuple
    if bounds is None:
        boundsTuple = (-np.inf, np.inf)
    else:
        boundsTuple = tuple(bounds)
    if args is None:
        args = []

    assert (
        isinstance(boundsTuple, tuple) and
        len(boundsTuple) == 2 and
        boundsTuple[1] > boundsTuple[0]
    ), "Derivative error: bounds must be a tuple of 2 elements or None."
    assert n in (0, 1, 2), "Derivative error: n must be 0, 1 or 2."
    assert order in (2, 4), "Derivative error: order must be 2 or 4."
    assert np.all(x <= boundsTuple[1]) and np.all(
        x >= boundsTuple[0]
    ), f"Derivative error: {x=} must be inside bounds."

    if n == 0:
        return f(x, *args)

    # If dx is not provided, we estimate it from scale and epsilon by minimizing
    # the total error ~ epsilon/dx**n + dx**order.
    dxFloat: float
    if dx is None:
        assert isinstance(epsilon, float), "Derivative error: epsilon must be a float."
        assert isinstance(scale, float), "Derivative error: scale must be a float."
        dxFloat = scale * epsilon ** (1 / (n + order))
    else:
        dxFloat = float(dx)

    # This step increases greatly the accuracy because it makes sure (x + dx) - x
    # is exactly equal to dx (no precision error).
    temp = x + dxFloat
    dxFloat = temp - x

    offset = np.zeros_like(x, dtype=int)
    offset -= x + dxFloat > float(boundsTuple[1])
    offset += x - dxFloat < float(boundsTuple[0])
    if order == 4:
        offset -= x + 2 * dxFloat > boundsTuple[1]
        offset += x - 2 * dxFloat < boundsTuple[0]

    if n == 1:
        pos = x[None, ...] + FIRST_DERIV_POS[str(order)].T[:, offset.tolist()]*dxFloat
        coeff = FIRST_DERIV_COEFF[str(order)].T[:, offset.tolist()] / dxFloat
    elif n == 2:
        pos = x[None, ...] + SECOND_DERIV_POS[str(order)].T[:, offset.tolist()]*dxFloat
        coeff = SECOND_DERIV_COEFF[str(order)].T[:, offset.tolist()] / dxFloat**2

    fx = f(pos, *args)
    fxShapeLength = len(fx.shape)
    coeffShapeLength = len(coeff.shape)
    return np.asarray(np.sum(
        coeff.reshape(coeff.shape + (fxShapeLength - coeffShapeLength) * (1,))
        * f(pos, *args),
        axis=0,
    ))


def gradient(
    f: Callable[..., np.ndarray],
    x: float | np.ndarray,
    order: int=4,
    epsilon: float=1e-16,
    scale: float | np.ndarray=1.0,
    dx: float | np.ndarray | None=None,
    axis: list | int | None=None,
    args: list | None=None,
) -> np.ndarray:
    r"""Computes the gradient of a callable function. Use the epsilon
    and scale parameters to estimate the optimal value of dx, if the latter is
    not provided.


    Parameters
    ----------
    f : function
        Function to differentiate. Should take an array as argument
        and return an array.
    x : array-like
        The position at which to evaluate the derivative. The size of the last
        axis must correspond to the number of variables on which f depends.
    order : int, optional
        The accuracy order of the scheme. Errors are of order
        :math:`\mathcal{O}({\rm d}x^{\text{order}/(\text{order}+1)})`.
        Can be 2 or 4. The default is 4.
    epsilon : float, optional
        Fractional accuracy at which f can be evaluated. If f is a simple
        function, should be close to the machine precision. Default is 1e-16.
    scale : float or array-like, optional
        Typical scale at which f(x) change by order 1. Can be an array, in which
        case each element corresponds to the scale of a different variable.
        Default is 1.
    dx : float or np.ndarray or None, optional
        The magnitude of finite differences. Can be an array, in which case
        each element corresponds to the dx of a different variable.If None, use
        epsilon and scale to estimate the optimal dx. Default is None.
    axis : list, int or None, optional
        Element of the gradient to return. If None, returns the whole gradient.
        Default is None.
    args: list, optional
        List of other fixed arguments passed to the function :math:`f`.

    Returns
    -------
    res : float
        The value of the gradient of :py:data:`f` evaluated at :py:data:`x`.

    """
    x = np.asarray(x)
    nbrVariables = x.shape[-1]

    if args is None:
        args = []

    axisList: list
    if isinstance(axis, int):
        axisList = [axis]
    elif axis is None:
        axisList = np.arange(nbrVariables).tolist()
    else:
        axisList = list(axis)
    for i in axisList:
        assert (
            -nbrVariables <= i < nbrVariables
        ), "Gradient error: axis must be between -nbrVariables and "\
           "nbrVariables-1 or None."

    assert order in (2,4), "Gradient error: order must be 2 or 4."

    # If dx is not provided, we estimate it from scale and epsilon by minimizing
    # the total error ~ epsilon/dx**n + dx**order.
    dxArray: np.ndarray
    if dx is None:
        assert isinstance(epsilon, float), "Gradient error: epsilon must be a float."

        if isinstance(scale, float):
            scale = scale * np.ones(nbrVariables)
        else:
            scale = np.asanyarray(scale)
            assert (
                scale.size == nbrVariables
            ), "Gradient error: scale must be a float or an array of size nbrVariables."
        dxArray = scale * epsilon ** (1 / (1 + order))
    elif isinstance(dx, float):
        dxArray = dx * np.ones(nbrVariables)
    else:
        dxArray = np.asarray(dx)
        assert (
            dxArray.size == nbrVariables
        ), "Gradient error: dx must be None, a float or an array of size nbrVariables."

    # This step increases greatly the accuracy because it makes sure (x + dx) - x
    # is exactly equal to dx (no precision error).
    temp = x + dxArray
    dxArray = temp - x

    pos = np.expand_dims(x, (-3, -2)) + FIRST_DERIV_POS[str(order)][
        0, :, None, None
    ] * np.identity(nbrVariables)[axisList, :] * np.expand_dims(dxArray, (-3, -2))
    shape = pos.shape[:-1]
    pos = pos.reshape((int(pos.size / nbrVariables), nbrVariables))
    coeff = FIRST_DERIV_COEFF[str(order)][0, :, None] / np.expand_dims(
        dxArray[..., axisList], -2
    )

    fEvaluation = f(pos, *args).reshape(shape)
    return np.asarray(np.sum(coeff * fEvaluation, axis=-2))


def hessian(
    f: Callable[..., np.ndarray],
    x: float | np.ndarray,
    order: int=4,
    epsilon: float=1e-16,
    scale: float | np.ndarray=1.0,
    dx: float | np.ndarray | None=None,
    xAxis: list | int | None=None,
    yAxis: list | int | None=None,
    args: list | None=None,
) -> np.ndarray:
    r"""Computes the hessian of a callable function. Use the epsilon
    and scale parameters to estimate the optimal value of dx, if the latter is
    not provided.


    Parameters
    ----------
    f : function
        Function to differentiate. Should take an array as argument
        and return an array.
    x : array-like
        The position at which to evaluate the derivative. The size of the last
        axis must correspond to the number of variables on which f depends.
    order : int, optional
        The accuracy order of the scheme. Errors are of order
        :math:`\mathcal{O}({\rm d}x^{\text{order}/(\text{order}+2)})`.
        Can be 2 or 4. The default is 4.
    epsilon : float, optional
        Fractional accuracy at which f can be evaluated. If f is a simple
        function, should be close to the machine precision. Default is 1e-16.
    scale : float, optional
        Typical scale at which f(x) change by order 1. Default is 1.
    dx : float or None, optional
        The magnitude of finite differences. If None, use epsilon and scale to
        estimate the optimal dx. Default is None.
    xAxis : list, int or None, optional
        Lines of the hessian matrix to return. If None, returns all the lines.
        Default is None.
    yAxis : list, int or None, optional
        Columns of the hessian matrix to return. If None, returns all the columns.
        Default is None.
    args: list, optional
        List of other fixed arguments passed to the function :math:`f`.

    Returns
    -------
    res : float
        The value of the hessian of :py:data:`f` evaluated at :py:data:`x`.

    """
    x = np.asarray(x)
    nbrVariables = x.shape[-1]

    if args is None:
        args = []

    xAxisList: list
    if isinstance(xAxis, int):
        xAxisList = [xAxis]
    elif xAxis is None:
        xAxisList = np.arange(nbrVariables).tolist()
    else:
        xAxisList = list(xAxis)
    for i in xAxisList:
        assert (
            -nbrVariables <= i < nbrVariables
        ), "Hessian error: axis must be between -nbrVariables and "\
           "nbrVariables-1 or None."
    yAxisList: list
    if isinstance(yAxis, int):
        yAxisList = [yAxis]
    elif yAxis is None:
        yAxisList = np.arange(nbrVariables).tolist()
    else:
        yAxisList = list(yAxis)
    for i in yAxisList:
        assert (
            -nbrVariables <= i < nbrVariables
        ), "Hessian error: axis must be between -nbrVariables and "\
           "nbrVariables-1 or None."

    assert order in (2, 4), "Hessian error: order must be 2 or 4."

    # If dx is not provided, we estimate it from scale and epsilon by minimizing
    # the total error ~ epsilon/dx**n + dx**order.
    dxArray: np.ndarray
    if dx is None:
        assert isinstance(epsilon, float), "Hessian error: epsilon must be a float."

        if isinstance(scale, float):
            scale = scale * np.ones(nbrVariables)
        else:
            scale = np.asanyarray(scale)
            assert (
                scale.size == nbrVariables
            ), "Hessian error: scale must be a float or an array of size nbrVariables."
        dxArray = scale * epsilon ** (1 / (2 + order))
    elif isinstance(dx, float):
        dxArray = dx * np.ones(nbrVariables)
    else:
        dxArray = np.asarray(dx)
        assert (
            dxArray.size == nbrVariables
        ), "Hessian error: dx must be None, a float or an array of size nbrVariables."

    # This step increases greatly the accuracy because it makes sure (x + dx) - x
    # is exactly equal to dx (no precision error).
    temp = x + dxArray
    dxArray = temp - x

    pos = (
        np.expand_dims(x, (-4, -3, -2))
        + HESSIAN_POS[str(order)][0, :, None, None, None]
        * np.identity(nbrVariables)[xAxisList, None, :]
        * np.expand_dims(dxArray, (-4, -3, -2))
        + HESSIAN_POS[str(order)][1, :, None, None, None]
        * np.identity(nbrVariables)[None, yAxisList, :]
        * np.expand_dims(dxArray, (-4, -3, -2))
    )
    shape = pos.shape[:-1]
    pos = pos.reshape((int(pos.size / nbrVariables), nbrVariables))
    coeff = HESSIAN_COEFF[str(order)][:, None, None] / (
        np.expand_dims(dxArray[..., yAxisList], (-3, -2))
        * np.expand_dims(dxArray[..., xAxisList], (-3, -1))
    )
    fEvaluation = f(pos, *args).reshape(shape)
    return np.asarray(np.sum(coeff * fEvaluation, axis=-3))


def gammaSq(v: float) -> float:
    r"""
    Lorentz factor :math:`\gamma^2` corresponding to velocity :math:`v`
    """
    return 1.0 / (1.0 - v * v)


def boostVelocity(xi: float, v: float) -> float:
    """
    Lorentz-transformed velocity
    """
    return (xi - v) / (1.0 - xi * v)


def nextStepDeton(
    pos1: float,
    pos2: float,
    pressure1: float,
    pressure2: float,
    mean2ndDeriv: float,
    std2ndDeriv: float,
    pressureTol: float,
    posMax: float,
    overshootProb: float = 0.05,
) -> float:
    """
    Function used in EquationOfMotion to find detonation solutions. It finds the next
    point to be sampled to try to bracket a root in such a way that the probability of
    overshooting a root is roughly equal to overshootProb.

    To estimate the overshoot probability, it fits the pressure to a quadratic which is
    equal to pressure2 at pos2, but with uncertain 1st and 2nd derivatives which are
    assumed to be normally distributed. The mean of the 1st derivative is computed by
    finite differences from the last 2 points.

    Parameters
    ----------
    pos1 : float
        Position of the first sampled point.
    pos2 : float
        position of the second sampled point.
    pressure1 : float
        Pressure at pos1.
    pressure2 : float
        Pressure at pos2.
    mean2ndDeriv : float
        Estimate of the 2nd derivative at pos2.
    std2ndDeriv : float
        Uncertainty on the 2nd derivative at pos2.
    pressureTol : float
        Relative accuracy at which pressure1 and pressure2 are computed.
    posMax : float
        Maximal position that the next step can have.
    overshootProb : float, optional
        Desired overshoot probability. A smaller value will lead to smaller step sizes
        which will take longer to evaluate, but with less chances of missing a root.
        The default is 0.05.

    Returns
    -------
    float
        Position where the overshoot probability is overshootProb (or posMax if there is
        no solution).

    """
    assert pos2 > pos1, "Error: pos2 must be greater than pos1."
    assert posMax > pos2, "Error: posMax must be greater than pos2."
    assert std2ndDeriv >= 0, "Error: std2ndDeriv must be positive."
    assert pressureTol >= 0, "Error: pressureTol must positive."
    assert 0 < overshootProb < 1, "Error: overshootProb must be between 0 and 1."

    # This function requires pressure2 to be negative. If that's not the case,
    # we invert the y axis.
    if pressure2 > 0:
        pressure1 *= -1
        pressure2 *= -1
        mean2ndDeriv *= -1

    # Use pressure units such that abs(pressure2)=1.
    # Helps when integrate to get the prob
    pressure1 /= abs(pressure2)
    mean2ndDeriv /= abs(pressure2)
    std2ndDeriv /= abs(pressure2)
    pressure2 = -1

    dx = pos2 - pos1
    dp = pressure2 - pressure1

    # Mean and variance of first derivative at pos2
    # (second term due to finite difference error)
    meanDeriv = dp / dx + dx * mean2ndDeriv / 2
    # First term: variance due to pressure uncertainty
    # Second term: variance due to finite difference error
    varDeriv = (pressure1**2 + pressure2**2) * (pressureTol / dx) ** 2 + (
        std2ndDeriv * dx / 2
    ) ** 2
    stdDeriv = np.sqrt(varDeriv)

    # Probability of overshooting a root when the 2nd derivative is positive:
    def probPositive(pos: float) -> float:
        if pos == pos2:
            return 0

        # Min derivative that can lead to overshooting
        def derivMin(secondDeriv: float) -> float:
            return -pressure2 / (pos - pos2) - secondDeriv * (pos - pos2) / 2

        # Probability density of the 2nd derivative
        def probDensity(secondDeriv: float) -> float:
            return float(np.exp(-((secondDeriv - mean2ndDeriv) ** 2) / std2ndDeriv**2/2)
            * special.erfc((derivMin(secondDeriv) - meanDeriv) / stdDeriv / np.sqrt(2))
            / (2 * std2ndDeriv * np.sqrt(2 * np.pi)))

        # Integrate the probability density to get the probability
        return float(integrate.quad(
            probDensity, -2 * pressure2 / (pos - pos2) ** 2, np.inf, full_output=1
        )[0])

    # Probability of overshooting 2 roots when the 2nd derivative is negative:
    def probNegative(pos: float) -> float:
        if pos == pos2:
            return 0

        # Min and max derivatives that can lead to overshooting
        def derivMin(secondDeriv: float) -> float:
            return float(np.sqrt(2 * secondDeriv * pressure2))
        def derivMax(secondDeriv: float) -> float:
            return -pressure2 / (pos - pos2) - secondDeriv * (pos - pos2) / 2

        # Probability density of the 2nd derivative
        def probDensity(secondDeriv: float) -> float:
            return float(np.exp(-((secondDeriv - mean2ndDeriv) ** 2) / std2ndDeriv**2/2)
            * (
                special.erf((derivMax(secondDeriv) - meanDeriv) / stdDeriv / np.sqrt(2))
                - special.erf(
                    (derivMin(secondDeriv) - meanDeriv) / stdDeriv / np.sqrt(2)
                )
            )
            / (2 * std2ndDeriv * np.sqrt(2 * np.pi))
        )

        # Integrate the probability density to get the probability
        return float(integrate.quad(
            probDensity, -np.inf, 2 * pressure2 / (pos - pos2) ** 2, full_output=1
        )[0])

    try:
        # Find a position for which the total probability is overshootProb
        return float(optimize.root_scalar(
            lambda pos: probPositive(pos) + probNegative(pos) - overshootProb,
            bracket=[pos2, posMax],
        ).root)
    except:
        # If no solution is found, returns posMax
        return posMax
