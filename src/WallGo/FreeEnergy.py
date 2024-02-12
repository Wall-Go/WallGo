import numpy as np
import numpy.typing as npt
import math
import scipy.integrate as scipyint

from .InterpolatableFunction import InterpolatableFunction
from .EffectivePotential import EffectivePotential
from .Fields import Fields


class FreeEnergyValueType(np.ndarray):

    def __new__(cls, arr: np.ndarray):
        obj = arr.view(cls)
        return obj

    def getVeffValue(self):
        """Returns value of the effective potential at a free-energy minimum.
        Returns a scalar if we only contain info for one temperature, otherwise returns a 1D array.
        """
        # Our last column is value of the potential at minimum.
        if (self.ndim < 2):
            values = self[-1]
        else:
            values = self[:, -1]
            if (len(values) == 1):
                values = values[0]

        return values

    def getFields(self):
        """Returns Fields array corresponding to local free energy minimum.
        """
        # Last column is Veff value, other columns are fields
        if (self.ndim < 2):
            values = self[:-1]
        else:
            values = self[:, :-1]
        return Fields.CastFromNumpy(values)


class FreeEnergy(InterpolatableFunction):
    """ Class FreeEnergy: Describes properties of a local effective potential minimum. 
    This is used to keep track of a minimum with respect to the temperature.
    By definition: free energy density of a phase == value of Veff in its local minimum.
    """

    effectivePotential: EffectivePotential
    # Approx field values where the phase lies at starting temperature
    startingTemperature: float
    startingPhaseLocationGuess: Fields

    # Lowest possible temperature so that the phase is still (meta)stable
    minPossibleTemperature: float
    # Highest possible temperature so that the phase is still (meta)stable
    maxPossibleTemperature: float

    def __init__(
        self,
        effectivePotential: EffectivePotential,
        startingTemperature: float,
        startingPhaseLocationGuess: Fields,
        initialInterpolationPointCount: int = 1000,
    ):

        adaptiveInterpolation = True
        # Set return value count. Currently the InterpolatableFunction requires this to be set manually:
        returnValueCount = startingPhaseLocationGuess.NumFields() + 1
        super().__init__(
            bUseAdaptiveInterpolation=adaptiveInterpolation,
            returnValueCount=returnValueCount,
            initialInterpolationPointCount=initialInterpolationPointCount,
        )

        self.effectivePotential = effectivePotential 
        self.startingTemperature = startingTemperature
        self.startingPhaseLocationGuess = startingPhaseLocationGuess

        self.minPossibleTemperature = 0.
        self.maxPossibleTemperature = np.Inf

    def __call__(self, x: npt.ArrayLike, useInterpolatedValues=True) -> FreeEnergyValueType:
        return FreeEnergyValueType( super().__call__(x, useInterpolatedValues) )

    def _functionImplementation(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """
        Parameters
        ----------
        temperature: float or numpy array of floats.
        """

        phaseLocation, potentialAtMinimum = self.effectivePotential.findLocalMinimum(self.startingPhaseLocationGuess, temperature)

        """We now need to make sure the field-independent but T-dependent contribution to free energy is included. 
        In principle this means we just call effectivePotential::evaluate().
        But here's a problem: currently if calling Veff with N field points and N temperatures, then numpy decideds to 
        produce a NxN array as a result. This means we end up doing unnecessary computations, and the resulting Veff values 
        are in wrong format!

        No solution currently, probably need to enforce correct broadcasting directly in Veff. As a hacky fix for the formatting I take the diagonal here.
        """
        # Actually the above seems to be fixed now, V1T was just implemented very badly. But leaving the comments here just in case

        potentialAtMinimum = np.real( self.effectivePotential.evaluate(phaseLocation, temperature) )

        """
        if (potentialAtMinimum.ndim > 1):
            potentialAtMinimum = np.diagonal(potentialAtMinimum).copy() # need to take a hard copy since np.diagonal gives just a read-only view
        """
            
        # Important: Minimization may not always work as intended, 
        # for example the minimum we're looking for may not even exist at the input temperature.
        # This will break interpolations unless we validate the result here. 
        # InterpolatableFunction is constructed to ignore inputs where the function evaluated to np.nan, 
        # so we avoid issues by returning np.nan here if the minimization failed.

        # How to do the validation? Perhaps the safest way would be to call this in a T-loop and storing the phaseLocation
        # at the previous T. If phaseLocation is wildly different at the next T, this may suggest that we ended up in a different minimum.
        # Issue with this approach is that it doesn't vectorize. Might not be a big loss however since findLocalMinimum itself is 
        # not effectively vectorized due to reliance on scipy routines.

        """TODO make the following work independently of how the Field array is organized.
        Too much hardcoded slicing right now."""

        # Here is a check that should catch "symmetry-breaking" type transitions where a field is 0 in one phase and nonzero in another
        bFieldWentToZero = (np.abs(self.startingPhaseLocationGuess) > 5.0) & (np.abs(phaseLocation) < 1e-1)

        # Check that we apply row-wise
        bEvaluationFailed = bFieldWentToZero # & ... add other checks ...

        # Make our failure check a boolean mask that numpy understands
        invalidRowMask = np.any(bEvaluationFailed, axis=1)

        # Replace all elements with np.nan on rows that failed the check
        phaseLocation[invalidRowMask, :] = np.nan
        potentialAtMinimum[invalidRowMask] = np.nan

        # reshape so that potentialAtMinimum is a column vector
        potentialAtMinimum_column = potentialAtMinimum[:, np.newaxis]

        # Join the arrays so that potentialAtMinimum is the last column and the others are as in phaseLocation
        result = np.concatenate((phaseLocation, potentialAtMinimum_column), axis=1)

        # This is now a 2D array where rows are [f1, f2, ..., Veff]
        return result

    def tracePhase(self, TMin: float, TMax: float, dT: float) -> None:
        """For now this will always update the interpolation table.
        """

        TMin = max(self.minPossibleTemperature, TMin)
        TMax = min(self.maxPossibleTemperature, TMax)

        numPoints = math.ceil((TMax-TMin) / dT)
        if not self.hasInterpolation():
            self.newInterpolationTable(TMin, TMax, numPoints)

        else:
            currentPoints = self.numPoints()
            self.extendInterpolationTable(TMin, TMax, math.ceil(numPoints / 2), math.ceil(currentPoints / 2))

        """We should now have interpolation table in range [TMin, TMax]. 
        If not, it suggests that our Veff minimization became invalid beyond some subrange [TMin', TMax']
        ==> Phase became unstable. 
        """
        if (self.interpolationRangeMax() < TMax):
            self.maxPossibleTemperature = self.interpolationRangeMax()

        if (self.interpolationRangeMin() > TMin):
            self.minPossibleTemperature = self.interpolationRangeMin()

    def tracePhaseIVP(self, TMin: float, TMax: float, dT: float, rTol: float, spinodal: bool) -> None:
        """
        Finds field(T) for the range over which it exists. Sets problem
        up as an initial value problem and uses scipy.integrate.solve_ivp to
        solve. Stops if we get sqrt(negative) or something like that.
        """
        # initial values
        T0 = self.startingTemperature
        phase0 = self.effectivePotential.findLocalMinimum(
            self.startingPhaseLocationGuess, T0,
        )
        ## HACK! a hard-coded tolerance
        tol_absolute = rTol * 0.1 * np.sqrt(T0)

        def ode_function(temperature, field):
            A = self.effectivePotential.ddV0(field, temperature)
            b = -self.effectivePotential.ddV0dT(field, temperature)
            return np.linalg.solve(A, b, assume_a="sym")

        # finding some sensible mass scales
        ddV_T0 = self.effectivePotential.ddV0(phase0, temperature)
        eigs_T0 = np.linalg.eigvalsh(ddV_T0)
        mass_scale_T0 = np.mean(eigs_T0)
        mass_hierarchy_T0 = min(eigs_T0) / max(eigs_T0)
        eps_test = rTol * 1e2
        
        def spinodal_event(temperature, field):
            # tests for if an eigenvalue of V'' goes through zero
            # or becomes very small compared to some initial mass scale
            # or if there is a large hierarchy in the eigenvalues
            A = self.ddV0(field, temperature)
            eigs = np.linalg.eigvalsh(A)
            test_zero = min(eigs)
            test_small = min(eigs) - eps_test * mass_scale_T0
            test_hierarchy = min(eigs) / max(eigs) - eps_test * mass_hierarchy_T0
            return min(test_zero, test_small, test_hierarchy)

        ode_up = scipyint.RK45(
            ode_function,
            T0,
            phase0,
            TMax,
            rtol=rTol,
            atol=tol_absolute,
            max_step=dT,
        )
        T_up = []
        field_up = []
        while ode_up.status == "running":
            try:
                ode_up.step()
            except RuntimeWarning as err:
                if err.args[0] != "invalid value encountered in sqrt":
                    raise
                else:
                    break
            if spinodal and spinodal_event(ode_up.t, ode_up.y) <= 0:
                break
            T_up.append(ode_up.t)
            field_up.append(ode_up.y)
            if ode_up.step_size < 0.01 * rTol * T0:
                break
        ode_down = scipyint.RK45(
            ode_function,
            T0,
            phase0,
            TMin,
            rtol=rTol,
            atol=tol_absolute,
            max_step=dT,
        )
        T_down = []
        field_down = []
        while ode_down.status == "running":
            try:
                ode_down.step()
            except RuntimeWarning as err:
                if err.args[0] != "invalid value encountered in sqrt":
                    raise
                else:
                    break
            if spinodal and spinodal_event(ode_up.t, ode_up.y) <= 0:
                break
            T_down.append(ode_down.t)
            field_down.append(ode_down.y)
            if ode_down.step_size < 0.01 * rTol * T0:
                break
        if len(T_down) <= 2:
            T_full = np.array(T_up)
            field_full = np.array(field_up)
        elif len(T_up) <= 2:
            T_full = np.flip(np.array(T_down), 0)
            field_full = np.flip(np.array(field_down), 0)
        else:
            T_full = np.append(np.flip(np.array(T_down), 0), np.array(T_up), 0)
            field_full = np.append(np.flip(np.array(field_down), 0), np.array(field_up), 0)

        # Now to construct the interpolation
        self.newInterpolationTableFromValues(T_full, field_full)
