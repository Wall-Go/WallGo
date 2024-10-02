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

import typing
import numpy as np
import scipy.integrate

from WallGo.interpolatableFunction import InterpolatableFunction, inputType, outputType


class JbIntegral(InterpolatableFunction):
    """
    Bosonic Jb(x), in practice use with x = m^2 / T^2.
    """

    SMALL_NUMBER: typing.Final[float] = 1e-100

    ## This doesn't vectorize nicely for numpy due to combination of piecewise
    ## scipy.integrate.quad and conditionals on x.
    # So for array input, let's just do a simple for loop
    def _functionImplementation(self, x: inputType | float) -> outputType:
        """
        Computes the bosonic one-loop thermal function Jb.

        Parameters
        ----------
        x : list[float] or np.ndarray or float
            Points where the funct`````ion will be evaluated.

        Returns
        -------
        list[float | np.ndarray] or np.ndarray
            Value of the thermal function

        """

        def wrapper(xWrapper: float) -> complex:
            """Wrapper for treating x>=0 and x<0 separately"""

            def integrandPositive(y: float) -> float:
                """This is the integrand if always y^2 + x >= 0. Taking abs of
                this and adding a small number inside log to help with numerics"""
                return float(
                    y**2
                    * np.log(
                        1.0
                        - np.exp(-np.sqrt(np.abs(y**2 + xWrapper)))
                        + self.SMALL_NUMBER
                    )
                )

            if xWrapper >= 0:

                resReal = scipy.integrate.quad(integrandPositive, 0.0, np.inf)[0]
                resImag = 0
            else:

                def integrandPrincipalLogReal(y: float) -> float:
                    """Now need to analytically continue, split the integral into
                    two parts: branch cut for y < sqrt(-x). Do some algebra to find
                    the principal log.
                    """
                    return float(
                        y**2
                        * np.log(
                            2 * np.abs(np.sin(0.5 * np.sqrt(-(y**2) - xWrapper)))
                            + self.SMALL_NUMBER
                        )
                    )

                def integrandPrincipalLogImag(y: float) -> float:
                    """Now need to analytically continue, split the integral into
                    two parts: branch cut for y < sqrt(-x). Here giving the result
                    for y^2 + x deformed slightly into the upper half complex plane.
                    """
                    return float(
                        y**2
                        * np.arctan(
                            1
                            / (
                                np.tan(0.5 * np.sqrt(-(y**2) - xWrapper))
                                + self.SMALL_NUMBER
                            )
                        )
                    )

                resReal = (
                    scipy.integrate.quad(
                        integrandPrincipalLogReal, 0.0, np.sqrt(np.abs(xWrapper))
                    )[0]
                    + scipy.integrate.quad(
                        integrandPositive, np.sqrt(np.abs(xWrapper)), np.inf
                    )[0]
                )
                resImag = scipy.integrate.quad(
                    integrandPrincipalLogImag, 0.0, np.sqrt(np.abs(xWrapper))
                )[0]

            return complex(resReal + 1j * resImag)

        if np.isscalar(x):
            res = wrapper(float(x))
            return np.asarray([[res.real, res.imag]])

        # one extra axis on x
        results = np.empty(np.asarray(x).shape + (2,), dtype=float)
        for i in np.ndindex(np.asarray(x).shape):
            res = wrapper(float(x[i]))
            results[i] = np.asarray([res.real, res.imag])

        return results


class JfIntegral(InterpolatableFunction):
    """
    Fermionic Jf(x), in practice use with x = m^2 / T^2. This is very similar to the
    bosonic counterpart Jb.
    """

    SMALL_NUMBER: typing.Final[float] = 1e-100

    def _functionImplementation(self, x: inputType | float) -> outputType:
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

        def wrapper(xWrapper: float) -> complex:

            def integrandPositive(y: float) -> float:
                """This is the integrand if always y^2 + x >= 0. Taking abs of this and
                adding a small number inside log to help with numerics."""
                return float(
                    -(y**2)
                    * np.log(
                        1 + np.exp(-np.sqrt((y**2) + xWrapper)) + self.SMALL_NUMBER
                    )
                )

            if xWrapper >= 0:
                resReal = scipy.integrate.quad(integrandPositive, 0.0, np.inf)[0]
                resImag = 0

            else:

                def integrandPrincipalLogReal(y: float) -> float:
                    """Principal log, similar to Jb."""
                    return float(
                        -(y**2)
                        * np.log(
                            2 * np.abs(np.cos(0.5 * np.sqrt(-(y**2) - xWrapper)))
                            + self.SMALL_NUMBER
                        )
                    )

                def integrandPrincipalLogImag(y: float) -> float:
                    """Imaginary part for y^2 + x deformed slightly into the upper
                    half complex plane."""
                    return float(
                        y**2
                        * np.arctan(
                            np.tan(0.5 * np.sqrt(-(y**2) - xWrapper))
                            + self.SMALL_NUMBER
                        )
                    )

                resReal = (
                    scipy.integrate.quad(
                        integrandPrincipalLogReal, 0.0, np.sqrt(np.abs(xWrapper))
                    )[0]
                    + scipy.integrate.quad(
                        integrandPositive, np.sqrt(np.abs(xWrapper)), np.inf
                    )[0]
                )
                resImag = scipy.integrate.quad(
                    integrandPrincipalLogImag, 0.0, np.sqrt(np.abs(xWrapper))
                )[0]

            # overall minus sign for Jf
            return complex(resReal + 1j * resImag)

        if np.isscalar(x):
            res = wrapper(float(x))
            return np.asarray([[res.real, res.imag]])

        # one extra axis on x
        results = np.empty(np.asarray(x).shape + (2,), dtype=float)
        for i in np.ndindex(np.asarray(x).shape):
            res = wrapper(float(x[i]))
            results[i] = np.asarray([res.real, res.imag])

        return results


class Integrals:
    """Class Integrals -- Just collects common integrals in one place.
    This is better than using global objects since in some cases
    we prefer their interpolated versions.
    """

    Jb: JbIntegral  # pylint: disable=invalid-name
    r"""Thermal 1-loop integral (bosonic):
        :math:`J_b(x) = \int_0^\infty dy y^2 \ln( 1 - \exp(-\sqrt(y^2 + x) ))`"""

    Jf: JfIntegral  # pylint: disable=invalid-name
    r""" Thermal 1-loop integral (fermionic):
        :math:`J_f(x) = -\int_0^\infty dy y^2 \ln( 1 + \exp(-\sqrt(y^2 + x) ))`"""

    def __init__(self) -> None:

        self.Jb = JbIntegral(  # pylint: disable=invalid-name
            bUseAdaptiveInterpolation=False
        )

        self.Jf = JfIntegral(  # pylint: disable=invalid-name
            bUseAdaptiveInterpolation=False
        )
