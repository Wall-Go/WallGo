"""
Class that does phase tracing, computes the effective potential in the minimum and
interpolate it.
"""

from dataclasses import dataclass
import logging
import numpy as np
import scipy.integrate as scipyint
import scipy.linalg as scipylinalg

from typing import Union

# from .containers import FreeEnergyArrays
from .effectivePotential import EffectivePotential
from .exceptions import WallGoError
from .fields import FieldPoint, Fields
from .interpolatableFunction import (
    InterpolatableFunction,
    EExtrapolationType,
    inputType,
    outputType,
)


@dataclass
class FreeEnergyValueType:
    """
    Data class containing the field value that minimizes the potential and the value of
    the potential in the minimum.
    """

    # Value of the effective potential at the free-energy minimum
    veffValue: np.ndarray
    # Values of background fields at the free-energy minimum
    fieldsAtMinimum: Fields

    @staticmethod
    def fromArray(arr: np.ndarray) -> "FreeEnergyValueType":
        """ASSUMES that the last column is Veff value."""
        # Awkward dimensionality check needed to figure out correct slicing
        if arr.ndim < 2:
            values = arr[-1]
            fields = arr[:-1]

        else:
            values = arr[:, -1]
            fields = arr[:, :-1]
            if len(values) == 1:
                values = values[0]

        return FreeEnergyValueType(
            veffValue=values, fieldsAtMinimum=Fields.castFromNumpy(fields)
        )


class FreeEnergy(InterpolatableFunction):
    """Class FreeEnergy: Describes properties of a local effective potential minimum.
    This is used to keep track of a minimum with respect to the temperature.
    By definition: free energy density of a phase == value of Veff in its local minimum.
    """

    def __init__(
        self,
        effectivePotential: EffectivePotential,
        startingTemperature: float,
        startingPhaseLocationGuess: Fields,
        initialInterpolationPointCount: int = 1000,
    ) -> None:
        """
        Initialize a FreeEnergy object

        Parameters
        ----------
        effectivePotential : EffectivePotential
            EffectivePotential object used to compute the free energy.
        startingTemperature: float
            Temperature at which the interpolation of the effective potential
            starts.
        startingPhaseLocationGuess : Fields
            Approximate position of the phase at startingTemperature.
        initialInterpolationPointCount : int, optional
            Initial number of points sampled for the interpolation.
            The default is 1000.

        """

        adaptiveInterpolation = True
        # Set return value count.
        # Currently the InterpolatableFunction requires this to be set manually:
        returnValueCount = startingPhaseLocationGuess.numFields() + 1
        super().__init__(
            bUseAdaptiveInterpolation=adaptiveInterpolation,
            returnValueCount=returnValueCount,
            initialInterpolationPointCount=initialInterpolationPointCount,
        )
        self.setExtrapolationType(EExtrapolationType.ERROR, EExtrapolationType.ERROR)

        self.effectivePotential = effectivePotential
        self.startingTemperature = startingTemperature
        # Approx field values where the phase lies at starting temperature
        self.startingPhaseLocationGuess = startingPhaseLocationGuess

        # List with lowest possible temperature for which the interpolated freeEnergy
        # is available, and bool which is true when the lowest possible temperature is
        # also the minimum temperature when the phase is still (meta)stable
        self.minPossibleTemperature = [0.0, False]
        # List with highest possible temperature for which the interpolated freeEnergy
        # is available, and bool which is true when the highest possible temperature is
        # also the maximum temperature when the phase is still (meta)stable
        self.maxPossibleTemperature = [np.inf, False]

    def evaluate(
        self, x: inputType, bUseInterpolatedValues: bool = True
    ) -> "FreeEnergyValueType":
        """
        Evaluate the free energy.

        Parameters
        ----------
        x : list[float] or np.ndarray
            Points where the free energy will be evaluated.
        bUseInterpolatedValues : bool, optional
            Whether or not to use interpolation to evaluate the function.
            The default is True.

        Returns
        -------
        FreeEnergyValueType
            FreeEnergyValueType object containing the value of the free energy and the
            field value that minimizes the potential.

        """
        # Implementation returns array, here we just unpack it.
        # Call to parent class needed to handle interpolation logic
        resultsArray = super().evaluate(x, bUseInterpolatedValues)
        return FreeEnergyValueType.fromArray(np.asarray(resultsArray))

    def __call__(
        self, x: inputType, bUseInterpolatedValues: bool = True
    ) -> "FreeEnergyValueType":
        """
        Evaluate the free energy.

        Parameters
        ----------
        x : list[float] or np.ndarray
            Points where the free energy will be evaluated.
        bUseInterpolatedValues : bool, optional
            Whether or not to use interpolation to evaluate the function.
            The default is True.

        Returns
        -------
        FreeEnergyValueType
            FreeEnergyValueType object containing the value of the free energy and the
            field value that minimizes the potential.

        """

        try:
            return self.evaluate(x, bUseInterpolatedValues)

        except ValueError:
            if self.maxPossibleTemperature[1] and self.minPossibleTemperature[1]:
                raise WallGoError(
                    "Trying to evaluate FreeEnergy outside of its range of existence"
                )
            raise WallGoError(
                """\n Trying to evaluate FreeEnergy outside of its allowed range,
                        try increasing/decreasing Tmax/Tmin."""
            )

    def _functionImplementation(self, temperature: inputType | float) -> outputType:
        """
        Internal implementation of free energy computation.
        You should NOT call this directly!
        Use the __call__() routine instead.

        Parameters
        ----------
        temperature: float or numpy array of floats.

        Returns
        -------
        freeEnergyArray: array-like
            Array with field values on the first columns and Veff values on the last
            column.
        """

        # InterpolatableFunction logic requires this to return things in array format,f
        # so we pack Veff(T) and minimum(T) into a 2D array. The __call__ routine above
        # unpacks this into a FreeEnergyValueType for easier use.
        # Hence you should NOT call this directly when evaluating free energy

        # Minimising potential. N.B. should already be real for this.
        phaseLocation, potentialAtMinimum = self.effectivePotential.findLocalMinimum(
            self.startingPhaseLocationGuess, temperature
        )

        # reshape so that potentialAtMinimum is a column vector
        potentialAtMinimumColumn = potentialAtMinimum[:, np.newaxis]

        # Join the arrays so that potentialAtMinimum is the last column and the others
        # are as in phaseLocation
        result = np.concatenate((phaseLocation, potentialAtMinimumColumn), axis=1)

        # This is now a 2D array where rows are [f1, f2, ..., Veff]
        return np.asarray(result)

    def derivative(
        self, x: inputType, order: int = 1, bUseInterpolation: bool = True
    ) -> "FreeEnergyValueType":
        """
        Override of InterpolatableFunction.derivative() function. Specifies accuracy
        based on our internal variables and puts the results in FreeEnergyValueType
        format. Otherwise similar to the parent function.

        Parameters
        ----------
        x : list[float] or np.ndarray
            Points where the free energy will be evaluated.
        bUseInterpolatedValues : bool, optional
            Whether or not to use interpolation to evaluate the function.
            The default is True.

        Returns
        -------
        FreeEnergyValueType
            FreeEnergyValueType object containing the value of the free energy's
            derivative and the field value that minimizes the potential.

        """
        resultsArray = super().derivative(
            x,
            order,
            bUseInterpolation=bUseInterpolation,
            epsilon=self.effectivePotential.getInherentRelativeError(),
            scale=self.effectivePotential.derivativeSettings.temperatureVariationScale,
        )

        return FreeEnergyValueType.fromArray(np.asarray(resultsArray))

    def tracePhase(
        self,
        TMin: float,
        TMax: float,
        dT: float,
        rTol: float = 1e-6,
        spinodal: bool = True,  # Stop tracing if a mass squared turns negative
        paranoid: bool = True,  # Re-solve minimum after every step
        phaseTracerFirstStep: float | None = None,  # Starting step
        interpolationDegree: int = 1,
    ) -> None:
        r"""Traces minimum of potential

        Finds field(T) for the range over which it exists. Takes a temperature
        derivative of the minimsation condition, and solves for :math:`\phi_i^\text{min}(T)` as an initial value problem

        .. math::
            \frac{\partial^2 V^\text{eff}}{\partial \phi_i \partial \phi_j}\bigg|_{\phi=\phi^\text{min}} \frac{\partial \phi^\text{min}_j}{\partial T} + \frac{\partial^2 V^\text{eff}}{\partial \phi_i \partial T}\bigg|_{\phi=\phi^\text{min}} = 0,

        starting from a solution at the starting temperature. It uses `scipy.integrate.solve_ivp` to solve the problem. Stops if a mass squared goes through zero.

        Parameters
        ----------
        TMin : float
            Minimal temperature at which the phase tracing will be done.
        TMax : float
            Maximal temperature at which the phase tracing will be done.
        dT : float
            Maximal temperature step size used by the phase tracer.
        rTol : float, optional
            Relative tolerance of the phase tracing. The default is :py:const:`1e-6`.
        spinodal : bool, optional
            If True, stop tracing if a mass squared turns negative. The default is True.
        paranoid : bool, optional
            If True, re-solve minimum after every step. The default is True.
        phaseTracerFirstStep : float or None, optional
            If a float, this gives the starting step size in units of the maximum step size :py:data:`dT`. If :py:data:`None` then uses the initial step size algorithm of :py:mod:`scipy.integrate.solve_ivp`. Default is :py:data:`None`
        interpolationDegree : int, optional
            Degree of the splines used in FreeEnergy to interpolate the potential and
            its derivatives. Default is 1.
        """
        # make sure the initial conditions are extra accurate
        extraTol = 0.01 * rTol

        # initial values, should be nice and accurate
        T0 = self.startingTemperature
        phase0Temp, potential0 = self.effectivePotential.findLocalMinimum(
            self.startingPhaseLocationGuess,
            T0,
            tol=extraTol,
        )
        phase0 = FieldPoint(phase0Temp[0])

        ## HACK! a hard-coded absolute tolerance
        tolAbsolute = rTol * max(*abs(phase0), T0)

        def odeFunction(temperature: float, field: np.ndarray) -> np.ndarray:
            # ode at each temp is a linear matrix equation A*x=b
            hess, dgraddT, _ = self.effectivePotential.allSecondDerivatives(
                FieldPoint(field), temperature
            )
            return np.asarray(scipylinalg.solve(hess, -dgraddT, assume_a="sym"))

        # compute all the second derivatives at the beginning of phase tracing
        d2Vdphi2, d2VdphidT, d2VdT2 = self.effectivePotential.allSecondDerivatives(
            phase0, T0)
        eigsT0 = np.linalg.eigvalsh(d2Vdphi2)

        # checking stable phase at initial temperature
        assert (
            min(eigsT0) > 0
        ), "tracePhase error: unstable at starting temperature"

        def spinodalEvent(temperature: float, field: np.ndarray) -> float:
            if not spinodal:
                return 1.0  # don't bother testing
            # tests for if an eigenvalue of V'' goes through zero
            d2V = self.effectivePotential.deriv2Field2(FieldPoint(field), temperature)
            eigs = scipylinalg.eigvalsh(d2V)
            return float(min(eigs))
                    

        # arrays to store results
        TList = np.full(1, T0)
        fieldList = np.full((1, phase0.numFields()), Fields((phase0,)))
        potentialEffList = np.full((1, 1), [potential0])
        dVdTList = np.full((1,1), [self.effectivePotential.derivT(phase0, T0)])
        dphidT = -np.linalg.inv(d2Vdphi2)@d2VdphidT
        d2VdT2List = np.full((1,1), [d2VdT2+dphidT@d2VdphidT])
        dphidTList = np.full((1, phase0.numFields()),
                             Fields((dphidT,)))
        # maximum temperature range
        TMin = max(self.minPossibleTemperature[0], TMin)
        TMax = min(self.maxPossibleTemperature[0], TMax)

        # kwargs for scipy.integrate.solve_ivp
        scipyKwargs = {
            "rtol": rTol,
            "atol": tolAbsolute,
            "max_step": dT,
            "first_step": phaseTracerFirstStep,
        }

        # iterating over up and down integration directions
        endpoints = [TMax, TMin]
        for direction in [0, 1]:
            TEnd = endpoints[direction]
            ode = scipyint.RK45(
                odeFunction,
                T0,
                phase0,
                TEnd,
                **scipyKwargs,
            )
            while ode.status == "running":
                try:
                    ode.step()
                    # check if all the eigenvalues of the hessian are positive
                    if spinodalEvent(ode.t, ode.y) <= 0:
                        break
                except RuntimeWarning as error:
                    logging.error(error.args[0] + f" at T={ode.t}")
                    break
                if paranoid:
                    phaset, potentialEffT = self.effectivePotential.findLocalMinimum(
                        Fields((ode.y)),
                        ode.t,
                        tol=rTol,
                    )
                    ode.y = phaset[0]
                else:
                    # check if extremum is still accurate
                    dVt = self.effectivePotential.derivField(Fields((ode.y)), ode.t)
                    err = np.linalg.norm(dVt) / T0**3
                    if err > rTol:
                        phaset, potentialEffT = (
                            self.effectivePotential.findLocalMinimum(
                                Fields((ode.y)),
                                ode.t,
                                tol=extraTol,
                            )
                        )
                        ode.y = phaset[0]
                    else:
                        # compute Veff
                        potentialEffT = np.asarray(
                            self.effectivePotential.evaluate(Fields((ode.y)), ode.t)
                        )

                # Computing all the derivatives along the whole phase tracing
                dVdT = self.effectivePotential.derivT(Fields((ode.y)), ode.t)
                (d2Vdphi2,
                 d2VdphidT,
                 d2VdT2) = self.effectivePotential.allSecondDerivatives(
                     FieldPoint(ode.y), ode.t)

                # check if step size is still okay to continue
                if ode.step_size < 1e-16 * T0 or (
                    TList.size > 0 and ode.t == TList[-1]
                ):
                    logging.warning(
                        "Step size %g shrunk too small at T=%g, vev=%g",
                        ode.step_size,
                        ode.t,
                        ode.y,
                    )
                    break

                dphidT = -np.linalg.inv(d2Vdphi2)@d2VdphidT
                D2VDT2 = d2VdT2+dphidT@d2VdphidT

                # check that sound speed square is still positive
                csq = dVdT/D2VDT2/ode.t
                if csq < 0:
                    break
                # Check if 2 methods for computing the 2nd derivative disagree by more
                # than a factor of 2. This would indicate a discontinuity caused by a
                # phase disappearing.
                if TList.size >= 2:
                    # The first method uses the last value stored in d2VdT2List, which
                    # computes the total derivative in terms of the partial derivatives
                    # of V and the field phi. This is the more accurate method.
                    # The second method takes the finite derivative of dVdT, which
                    # should break down when the phase disappears because there is a
                    # discontinuity. 
                    if d2VdT2List[-1,0]*(ode.t-TList[-2])/(dVdT-dVdTList[-2,0]) < 0.5:
                        break

                # append results to lists
                TList = np.append(TList, [ode.t], axis=0)
                fieldList = np.append(fieldList, [ode.y], axis=0)
                potentialEffList = np.append(potentialEffList, [potentialEffT], axis=0)
                dVdTList = np.append(dVdTList, [[dVdT]], axis=0)
                d2VdT2List = np.append(d2VdT2List, [[D2VDT2]], axis=0)
                dphidTList = np.append(dphidTList,
                                       [dphidT],
                                       axis=0)
            if direction == 0:
                # populating results array
                TFullList = TList
                fieldFullList = fieldList
                potentialEffFullList = potentialEffList
                dVdTFullList = dVdTList
                d2VdT2FullList = d2VdT2List
                dphidTFullList = dphidTList
                # making new empty array for downwards integration
                TList = np.empty(0, dtype=float)
                fieldList = np.empty((0, phase0.numFields()), dtype=float)
                potentialEffList = np.empty((0, 1), dtype=float)
                dVdTList = np.empty((0, 1), dtype=float)
                d2VdT2List = np.empty((0, 1), dtype=float)
                dphidTList = np.empty((0, phase0.numFields()), dtype=float)
            else:
                if len(TList) > 1:
                    # combining up and down integrations
                    TFullList = np.append(np.flip(TList, 0), TFullList, axis=0)
                    fieldFullList = np.append(
                        np.flip(fieldList, axis=0), fieldFullList, axis=0
                    )
                    potentialEffFullList = np.append(
                        np.flip(potentialEffList, axis=0), potentialEffFullList, axis=0
                    )
                    dVdTFullList = np.append(
                        np.flip(dVdTList, axis=0), dVdTFullList, axis=0)
                    d2VdT2FullList = np.append(
                        np.flip(d2VdT2List, axis=0), d2VdT2FullList, axis=0)
                    dphidTFullList = np.append(
                        np.flip(dphidTList, axis=0), dphidTFullList, axis=0
                    )
                elif len(TFullList) <= 1:
                    # Both up and down lists are too short
                    raise RuntimeError("Failed to trace phase")

        # overwriting temperature range
        ## HACK! Hard-coded 2*dT, see issue #145
        self.minPossibleTemperature[0] = min(TFullList) + 2 * dT
        self.maxPossibleTemperature[0] = max(TFullList) - 2 * dT
        assert (
            self.maxPossibleTemperature > self.minPossibleTemperature
        ), f"Temperature range negative: decrease dT from {dT}"

        if min(TFullList) > TMin:
            self.minPossibleTemperature[1] = True

        if max(TFullList) < TMax:
            self.maxPossibleTemperature[1] = True

        if (
            self.maxPossibleTemperature[0]
            < ode.step_size * 10 + self.startingTemperature
            or self.minPossibleTemperature[0]
            > self.startingTemperature - ode.step_size * 10
        ):
            logging.warning(
                """Warning: the temperature step size seems too large.
                Try decreasing temperatureVariationScale."""
            )

        # Compute the second derivative of the field by finite differences
        d2phidT2 = np.zeros_like(dphidTFullList)
        d2phidT2[1:-1] = ((dphidTFullList[2:] - dphidTFullList[:-2]) /
                          (TFullList[2:] - TFullList[:-2])[:,None])
        d2phidT2[0] = ((dphidTFullList[1] - dphidTFullList[0]) /
                          (TFullList[1] - TFullList[0]))
        d2phidT2[-1] = ((dphidTFullList[-1] - dphidTFullList[-2]) /
                          (TFullList[-1] - TFullList[-2]))

        # Now to construct the interpolation
        result = np.concatenate((fieldFullList, potentialEffFullList), axis=1)
        deriv1 = np.concatenate((dphidTFullList, dVdTFullList), axis=1)
        deriv2 = np.concatenate((d2phidT2, d2VdT2FullList), axis=1)
        self.newInterpolationTableFromValues(TFullList, result, [deriv1, deriv2],
                                             interpolationDegree)

    def constructInterpolationFromArray(
        self,
        freeEnergyArrays: "FreeEnergyArrays",
        dT: float,
    ) -> None:
        """
        Constructs the interpolation table directly from arrays of temperatures,
        field values, and potential values, bypassing the phase tracing process.

        Parameters
        ----------
        freeEnergyArrays : FreeEnergyArrays
            Object containing arrays of the temperature, minimum and value of the free energy.
        dT : float
            Small step in temperature used in derivatives, used here to ensure endpoints of temperature range not exceeded when taking derivatives later.
        """
        if freeEnergyArrays.allowedDiscrepancy is None:
            freeEnergyArrays.allowedDiscrepancy = self.effectivePotential.effectivePotentialError

        freeEnergyList = freeEnergyArrays.freeEnergyList

        # Check if the loaded value is consistent with the effective potential
        discrepancies = abs(
            freeEnergyList.veffValue - self.effectivePotential.evaluate(
                freeEnergyList.fieldsAtMinimum, freeEnergyArrays.temperatures
            )
        )
        # Norm includes neighbouring points to avoid division by zero
        discrepancyNorm = np.concatenate(
            ([0.5 * (abs(freeEnergyList.veffValue[0]) + abs(freeEnergyList.veffValue[1]))],
            0.5 * (abs(freeEnergyList.veffValue[:-1]) + abs(freeEnergyList.veffValue[1:])))
        )
        maxDiscrepancy = max(discrepancies / discrepancyNorm)
        if (maxDiscrepancy > freeEnergyArrays.allowedDiscrepancy):
            raise WallGoError(
                f"The loaded phase disagrees with the effective potential at {maxDiscrepancy:g}, higher than the required precision of {freeEnergyArrays.allowedDiscrepancy:g}."
            )

        # Check that the provided array has sufficiently small temperature steps
        maxTemperatureStep = max(abs(
            freeEnergyArrays.temperatures[:-1]-freeEnergyArrays.temperatures[1:]
        ))

        if maxTemperatureStep > dT:
            logging.warning(
                "The maximum temperature step size %g is larger than the maximum step size %g. "
                "This may lead to inaccurate interpolation.",
                maxTemperatureStep,
                dT,
            )

        self.minPossibleTemperature[0] = min(freeEnergyArrays.temperatures) + 2 * dT
        self.maxPossibleTemperature[0] = max(freeEnergyArrays.temperatures) - 2 * dT

        # Concatenate field values and potential into a single array (N, nFields + 1)
        resultArray = np.concatenate(
            (freeEnergyList.fieldsAtMinimum, freeEnergyList.veffValue[:, np.newaxis]), axis=1
        )

        # Construct the interpolation table
        self.newInterpolationTableFromValues(
            freeEnergyArrays.temperatures, resultArray
        )
