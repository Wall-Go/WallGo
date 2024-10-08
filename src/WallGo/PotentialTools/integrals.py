r"""
One-loop thermal integrals used to compute the effective potential.

For 1-loop thermal potential WITHOUT high-T approximations, need to calculate

    .. math::
        J_b(x) =  \int_0^\infty dy y^2 \ln( 1 - \exp(-\sqrt(y^2 + x) )) \text{ (bosonic),}

        J_f(x) = -\int_0^\infty dy y^2 \ln( 1 + \exp(-\sqrt(y^2 + x) )) \text{ (fermionic)}.

The thermal 1-loop correction from one particle species with N degrees of freedom is
then

.. math::
    V_1(T) = T^4/(2\pi^2) N J(m^2 / T^2).

See e.g. CosmoTransitions (arXiv:1109.4189, eq. (12)). 

Particularly for scalars the :math:`m^2` can be negative so we allow :math:`x < 0`, 
but we calculate the real parts of integrals only. 
NB: for large negative :math:`x` the integrals are slow to compute and good convergence
is not guaranteed by the quad integrator used here.

Note also that the while the analytical continuation to :math:`x < 0` makes sense
mathematically, it is physically less clear whether this is the right thing to use.
Here we just provide implementations of :math:`J_b(x)` and :math:`J_f(x)`; it is up to the user to
decide how to deal with negative input.

Usage: We define :py:class:`JbIntegral` and :py:class:`JbIntegral` are defined as :py:class:`InterpolatableFunction` to allow optimized
evaluation. The individual integrals are then collected in the :py:class:`Integrals` class
below. `WallGo` provides a default Integrals object defined in WallGo's :py:mod:`__init__.py`,
accessible as :py:data:`WallGo.defaultIntegrals`. Once :py:meth:`WallGo.initialize()` is called, we optimize
Jb and Jf in :py:data:`WallGo.defaultIntegrals` by loading their interpolation tables. 
"""

import typing
import numpy as np
import scipy.integrate

from ..interpolatableFunction import InterpolatableFunction, inputType, outputType


def _integrator(
    func: typing.Callable, a: float, b: float
) -> float:
    """
    Simple wrapper for scipy.integrate.quad with defaults inbuilt
    """
    res = scipy.integrate.quad(
        func,
        a,
        b,
        limit=100,
    )
    return float(res[0])


class JbIntegral(InterpolatableFunction):
    """
    Bosonic Jb(x), in practice use with x = m^2 / T^2.
    """

    SMALL_NUMBER: typing.Final[float] = 1e-100

    def __init__(
        self,
        bUseAdaptiveInterpolation: bool = True,
        initialInterpolationPointCount: int = 1000,
        returnValueCount: int = 2,
    ) -> None:
        super().__init__(
            bUseAdaptiveInterpolation,
            initialInterpolationPointCount,
            returnValueCount,
        )

    @classmethod
    def _integrandPositiveReal(cls, x: float, y: float) -> float:
        """
        The real part of integrand of the Jb function, for positive y.
        Note the imaginary part for positive y is just zero.
        """
        return float(
            y**2 * np.log(
                1.0 - np.exp(-np.sqrt(y**2 + x)) + cls.SMALL_NUMBER
            )
        )

    @classmethod
    def _integrandNegativeReal(cls, x: float, y: float) -> float:
        """
        The (principal) real part of integrand of the Jb function, for negative y,
        slighly deformed into the y upper-half plane.
        """
        return float(
            y**2 * np.log(
                2 * np.abs(np.sin(0.5 * np.sqrt(-y**2 - x))) + cls.SMALL_NUMBER
            )
        )

    @classmethod
    def _integrandNegativeImaginary(cls, x: float, y: float) -> float:
        """
        The imaginary part of integrand of the Jb function, for negative y,
        slighly deformed into the y upper-half plane.
        """
        return float(
            y**2 * np.arctan(
                1 / (
                    np.tan(0.5 * np.sqrt(-y**2 - x)) + cls.SMALL_NUMBER
                )
            )
        )

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

            if xWrapper >= 0:
                resReal = _integrator(
                    lambda y: JbIntegral._integrandPositiveReal(xWrapper, y),
                    0.0,
                    np.inf,
                )
                resImag = 0.0
            else:
                resReal = (
                    _integrator(
                        lambda y: JbIntegral._integrandNegativeReal(xWrapper, y),
                        0.0,
                        np.sqrt(np.abs(xWrapper))
                    )
                    + _integrator(
                        lambda y: JbIntegral._integrandPositiveReal(xWrapper, y),
                        np.sqrt(np.abs(xWrapper)),
                        np.inf
                    )
                )
                resImag = _integrator(
                    lambda y: JbIntegral._integrandNegativeImaginary(xWrapper, y),
                    0.0,
                    np.sqrt(np.abs(xWrapper))
                )

            return complex(resReal + 1j * resImag)

        if np.isscalar(x):
            res = wrapper(float(x))
            return np.asarray([res.real, res.imag])

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

    def __init__(
        self,
        bUseAdaptiveInterpolation: bool = True,
        initialInterpolationPointCount: int = 1000,
        returnValueCount: int = 2,
    ) -> None:
        super().__init__(
            bUseAdaptiveInterpolation,
            initialInterpolationPointCount,
            returnValueCount,
        )

    @classmethod
    def _integrandPositiveReal(cls, x: float, y: float) -> float:
        """
        The real part of integrand of the Jb function, for positive y.
        Note the imaginary part for positive y is just zero.
        """
        return float(
            -y**2 * np.log(
                1 + np.exp(-np.sqrt((y**2) + x)) + cls.SMALL_NUMBER
            )
        )

    @classmethod
    def _integrandNegativeReal(cls, x: float, y: float) -> float:
        """
        The (principal) real part of integrand of the Jb function, for negative y,
        slighly deformed into the y upper-half plane.
        """
        return float(
            -y**2 * np.log(
                2 * np.abs(np.cos(0.5 * np.sqrt(-y**2 - x))) + cls.SMALL_NUMBER
            )
        )

    @classmethod
    def _integrandNegativeImaginary(cls, x: float, y: float) -> float:
        """
        The imaginary part of integrand of the Jb function, for negative y,
        slighly deformed into the y upper-half plane.
        """
        return float(
            y**2 * np.arctan(
                np.tan(0.5 * np.sqrt(-(y**2) - x)) + cls.SMALL_NUMBER
            )
        )

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
            """Wrapper for treating x>=0 and x<0 separately"""

            if xWrapper >= 0:
                resReal = _integrator(
                    lambda y: JfIntegral._integrandPositiveReal(xWrapper, y),
                    0.0,
                    np.inf,
                )
                resImag = 0.0
            else:
                resReal = (
                    _integrator(
                        lambda y: JfIntegral._integrandNegativeReal(xWrapper, y),
                        0.0,
                        np.sqrt(np.abs(xWrapper))
                    )
                    + _integrator(
                        lambda y: JfIntegral._integrandPositiveReal(xWrapper, y),
                        np.sqrt(np.abs(xWrapper)),
                        np.inf
                    )
                )
                resImag = _integrator(
                    lambda y: JfIntegral._integrandNegativeImaginary(xWrapper, y),
                    0.0,
                    np.sqrt(np.abs(xWrapper))
                )

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
