import numpy as np
import numpy.typing as npt
import math

from .InterpolatableFunction import InterpolatableFunction
from .EffectivePotential import EffectivePotential
from .Fields import Fields

class FreeEnergyValueType(np.ndarray):
        
    def __new__(cls, arr: np.ndarray):
        obj = arr.view(cls)
        return obj
    
    def getVeffValue(self):
        ## Our last column is value of the potential at minimum
        return self[:, -1]


class FreeEnergy(InterpolatableFunction):
    """ Class FreeEnergy: Describes properties of a local effective potential minimum. 
    This is used to keep track of a minimum with respect to the temperature.
    By definition: free energy density of a phase == value of Veff in its local minimum.
    """

    effectivePotential: EffectivePotential
    ## Approx field values where the phase lies (TODO should we include T-dependence?)
    phaseLocationGuess: Fields

    minPossibleTemperature: float ## Lowest possible temperature so that the phase is still (meta)stable 
    maxPossibleTemperature: float ## Highest possible temperature so that the phase is still (meta)stable

    def __init__(self, effectivePotential: EffectivePotential, phaseLocationGuess: Fields, initialInterpolationPointCount: int=1000):

        adaptiveInterpolation = True
        ## Set return value count. Currently the InterpolatableFunction requires this to be set manually:
        returnValueCount = phaseLocationGuess.NumFields() + 1
        super().__init__(bUseAdaptiveInterpolation=adaptiveInterpolation, returnValueCount=returnValueCount, initialInterpolationPointCount=initialInterpolationPointCount)

        self.effectivePotential = effectivePotential 
        self.phaseLocationGuess = phaseLocationGuess

        self.minPossibleTemperature = 0.
        self.maxPossibleTemperature = np.Inf


    def _functionImplementation(self, temperature: npt.ArrayLike) -> FreeEnergyValueType:
        """
        Parameters
        ----------
        temperature: float or numpy array of floats.
        """

        phaseLocation, potentialAtMinimum = self.effectivePotential.findLocalMinimum(self.phaseLocationGuess, temperature)

        ## Make sure the field-independent but T-dependent contribution to free energy is included
        potentialAtMinimum = np.real( self.effectivePotential.evaluateWithConstantPart(phaseLocation, temperature) )

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
        bFieldWentToZero = (np.abs(self.phaseLocationGuess) > 5.0) & (np.abs(phaseLocation) < 1e-1)

        ## Check that we apply row-wise
        bEvaluationFailed = bFieldWentToZero ## & ... add other checks ...

        ## Make our failure check a boolean mask that numpy understands
        invalidRowMask = np.any(bEvaluationFailed, axis=1)

        ## Replace all elements with np.nan on rows that failed the check
        phaseLocation[invalidRowMask, :] = np.nan
        potentialAtMinimum[invalidRowMask] = np.nan

        # reshape so that potentialAtMinimum is a column vector
        potentialAtMinimum_column = potentialAtMinimum[:, np.newaxis]

        # Join the arrays so that potentialAtMinimum is the last column and the others are as in phaseLocation
        result = np.concatenate((phaseLocation, potentialAtMinimum_column), axis=1)

        ## This is now a 2D array where rows are [f1, f2, ..., Veff]
        return FreeEnergyValueType(result)
    


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

