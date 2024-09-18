r"""
One-loop thermal integrals used to compute the effective potential.

For 1-loop thermal potential WITHOUT high-T approximations, need to calculate
:math:`J_b(x) =  \int_0^\infty dy y^2 \ln( 1 - \exp(-\sqrt(y^2 + x) ))` (bosonic)
and 
:math:`J_f(x) = -\int_0^\infty dy y^2 \ln( 1 + \exp(-\sqrt(y^2 + x) ))` (fermionic).
The thermal 1-loop correction from one particle species with N degrees of freedom is
then :math:`V_1(T) = T^4/(2\pi^2) N J(m^2 / T^2)`.
See eg. CosmoTransitions (arXiv:1109.4189, eq. (12)). 

Particularly for scalars the m^2 can be negative so we allow x < 0, 
but we calculate the real parts of integrals only. 
NB: for large negative x the integrals are slow to compute and good convergence
is not guaranteed by the quad integrator used here.

Note also that the while the analytical continuation to x < 0 makes sense
mathematically, it is physically less clear whether this is the right thing to use.
Here we just provide implementations of J_b(x) and J_f(x); it is up to the user to
decide how to deal with negative input.

Usage: We define Jb and Jf are defined as InterpolatableFunction to allow optimized
evaluation. The individual integrals are then collected in the ``Integrals`` class
below. WallGo provides a default Integrals object defined in WallGo's __init__.py,
accessible as WallGo.defaultIntegrals. Once WallGo.initialize() is called, we optimize
Jb and Jf in WallGo.defaultIntegrals by loading their interpolation tables. 
"""

import numpy as np
import scipy.integrate

from .interpolatableFunction import InterpolatableFunction

inputType = list[float] | np.ndarray | float
outputType = list[float | np.ndarray] | np.ndarray

class JbIntegral(InterpolatableFunction):
    """
    Bosonic Jb(x), in practice use with x = m^2 / T^2.
    """

    SMALL_NUMBER = 1e-100 # pylint: disable=invalid-name

    ## This doesn't vectorize nicely for numpy due to combination of piecewise
    ## scipy.integrate.quad and conditionals on x.
    # So for array input, let's just do a simple for loop
    def _functionImplementation(self, x: inputType) -> outputType:
        """
        Computes the bosonic one-loop thermal function Jb.

        Parameters
        ----------
        x : list[float] or np.ndarray or float
            Points where the function will be evaluated.

        Returns
        -------
        list[float | np.ndarray] or np.ndarray
            Value of the thermal function

        """

        def wrapper(xWrapper: float) -> float:
            # This is the integrand if always y^2 + x >= 0. Taking abs of this and
            # adding a small number inside log to help with numerics

            def integrand(y: float) -> float:
                return float(y**2*np.log(1.0 - np.exp(-np.sqrt(np.abs(y**2 + xWrapper)))
                                      + self.SMALL_NUMBER))

            if xWrapper >= 0:
                res = scipy.integrate.quad(integrand, 0.0, np.inf)[0]

            else:
                ## Now need to analytically continue, split the integral into two parts
                ## (complex for y < sqrt(-x))
                # Do some algebra to find the principal log (we do real parts only)
                def integrandPrincipalLog(y: float) -> float:
                    return float(y * y * np.log(
                        2.0 * np.abs(np.sin(0.5 * np.sqrt(np.abs(y * y + xWrapper))))
                        + self.SMALL_NUMBER))

                res = (
                    scipy.integrate.quad(
                        integrandPrincipalLog, 0.0, np.sqrt(np.abs(xWrapper))
                    )[0]
                    + scipy.integrate.quad(integrand,
                                           np.sqrt(np.abs(xWrapper)), np.inf)[0]
                )
            return float(res.real)

        if np.isscalar(x):
            return np.asarray(wrapper(float(x)))

        results = np.empty_like(x, dtype=float)  # same shape as x
        for i in np.ndindex(np.asarray(x).shape):
            results[i] = wrapper(x[i])

        return results


class JfIntegral(InterpolatableFunction):
    """
    Fermionic Jf(x), in practice use with x = m^2 / T^2. This is very similar to the
    bosonic counterpart Jb.
    """

    SMALL_NUMBER = 1e-100 # pylint: disable=invalid-name

    def _functionImplementation(self, x: inputType) -> outputType:
        """
        Computes the fermionic one-loop thermal function Jf.

        Parameters
        ----------
        x : list[float] or np.ndarray or float
            Points where the function will be evaluated.

        Returns
        -------
        list[float | np.ndarray] or np.ndarray
            Value of the thermal function

        """

        def wrapper(xWrapper: float) -> float:
            # This is the integrand if always y^2 + x >= 0. Taking abs of this and
            # adding a small number inside log to help with numerics
            def integrand(y: float) -> float:
                return float(y**2*np.log(1.0 + np.exp(-np.sqrt(np.abs(y**2 + xWrapper)))
                                      + self.SMALL_NUMBER))

            if xWrapper >= 0:
                res = scipy.integrate.quad(integrand, 0.0, np.inf)[0]

            else:
                # Like Jb but now we get a cos
                def integrandPrincipalLog(y: float) -> float:
                    return float(y * y * np.log(
                        2.0 * np.abs(np.cos(0.5 * np.sqrt(np.abs(y * y + xWrapper))))
                        + self.SMALL_NUMBER))

                res = (
                    scipy.integrate.quad(
                        integrandPrincipalLog, 0.0, np.sqrt(np.abs(xWrapper))
                    )[0]
                    + scipy.integrate.quad(integrand,
                                           np.sqrt(np.abs(xWrapper)), np.inf)[0]
                )

            # overall minus sign for Jf
            return -float(res.real)

        if np.isscalar(x):
            return np.asarray(wrapper(float(x)))

        results = np.empty_like(x, dtype=float)  # same shape as x
        for i in np.ndindex(np.asarray(x).shape):
            results[i] = wrapper(x[i])

        return results


class Integrals:
    """Class Integrals -- Just collects common integrals in one place.
    This is better than using global objects since in some cases
    we prefer their interpolated versions.
    """

    Jb: JbIntegral # pylint: disable=invalid-name
    Jf: JfIntegral # pylint: disable=invalid-name

    def __init__(self) -> None:

        r"""Thermal 1-loop integral (bosonic):
            :math:`J_b(x) = \int_0^\infty dy y^2 \ln( 1 - \exp(-\sqrt(y^2 + x) ))`"""
        self.Jb = JbIntegral( # pylint: disable=invalid-name
            bUseAdaptiveInterpolation=False)

        r""" Thermal 1-loop integral (fermionic):
            :math:`J_f(x) = -\int_0^\infty dy y^2 \ln( 1 + \exp(-\sqrt(y^2 + x) ))`"""
        self.Jf = JfIntegral( # pylint: disable=invalid-name
            bUseAdaptiveInterpolation=False)
