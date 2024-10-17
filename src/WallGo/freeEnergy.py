"""
Class that does phase tracing, computes the effective potential in the minimum and
interpolate it.
"""

from dataclasses import dataclass
import logging
import numpy as np
import scipy.integrate as scipyint
import scipy.linalg as scipylinalg

from .interpolatableFunction import (
    InterpolatableFunction,
    EExtrapolationType,
    inputType,
    outputType,
)
from .effectivePotential import EffectivePotential
from .exceptions import WallGoError
from .fields import FieldPoint, Fields


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
    ) -> None:
        """
        Finds field(T) for the range over which it exists. Sets problem
        up as an initial value problem and uses scipy.integrate.solve_ivp to
        solve. Stops if we get sqrt(negative) or something like that.

        Parameters
        ----------
        TMin : float
            Minimal temperature at which the phase tracing will be done.
        TMax : float
            Maximal temperature at which the phase tracing will be done.
        dT : float
            Maximal temperature step size used by the phase tracer.
        rTol : float, optional
            Relative tolerance of the phase tracing. The default is 1e-6.
        spinodal : bool, optional
            If True, stop tracing if a mass squared turns negative. The default is True.
        paranoid : bool, optional
            If True, re-solve minimum after every step. The default is True.

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

        # finding some sensible mass scales
        ddVT0 = self.effectivePotential.deriv2Field2(phase0, T0)
        eigsT0 = np.linalg.eigvalsh(ddVT0)
        # mass_scale_T0 = np.mean(eigs_T0)
        # min_mass_scale = rTol * mass_scale_T0
        # mass_hierarchy_T0 = min(eigs_T0) / max(eigs_T0)
        # min_hierarchy = rTol * mass_hierarchy_T0

        # checking stable phase at initial temperature
        assert (
            min(eigsT0) * max(eigsT0) > 0
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

        # maximum temperature range
        TMin = max(self.minPossibleTemperature[0], TMin)
        TMax = min(self.maxPossibleTemperature[0], TMax)

        # iterating over up and down integration directions
        endpoints = [TMax, TMin]
        for direction in [0, 1]:
            TEnd = endpoints[direction]
            ode = scipyint.RK45(
                odeFunction,
                T0,
                phase0,
                TEnd,
                rtol=rTol,
                atol=tolAbsolute,
                max_step=dT,
            )
            while ode.status == "running":
                try:
                    ode.step()
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
                if spinodalEvent(ode.t, ode.y) <= 0:
                    break
                if not paranoid:
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
                # check if step size is still okay to continue
                if ode.step_size < 1e-16 * T0 or (
                    TList.size > 0 and ode.t == TList[-1]
                ):
                    logging.warning(
                        f"Step size {ode.step_size} shrunk too small at T={ode.t}, "
                        f"vev={ode.y}"
                    )
                    break
                # append results to lists
                TList = np.append(TList, [ode.t], axis=0)
                fieldList = np.append(fieldList, [ode.y], axis=0)
                potentialEffList = np.append(potentialEffList, [potentialEffT], axis=0)
            if direction == 0:
                # populating results array
                TFullList = TList
                fieldFullList = fieldList
                potentialEffFullList = potentialEffList
                # making new empty array for downwards integration
                TList = np.empty(0, dtype=float)
                fieldList = np.empty((0, phase0.numFields()), dtype=float)
                potentialEffList = np.empty((0, 1), dtype=float)
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

        # Now to construct the interpolation
        result = np.concatenate((fieldFullList, potentialEffFullList), axis=1)
        self.newInterpolationTableFromValues(TFullList, result)
