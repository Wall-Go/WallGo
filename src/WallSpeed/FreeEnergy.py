import numpy as np
import numpy.typing as npt

from .InterpolatableFunction import InterpolatableFunction
from .EffectivePotential import EffectivePotential

""" Class FreeEnergy: Describes properties of a local effective potential minimum. 
This is used to keep track of a minimum with respect to the temperature.
By definition: free energy density of a phase == value of Veff in its local minimum.
"""
class FreeEnergy(InterpolatableFunction):

    def __init__(self, effectivePotential: EffectivePotential, phaseLocationGuess: list[float]):

        adaptiveInterpolation = True
        ## Set return value count. Currently the InterpolatableFunction requires this to be set manually:
        returnValueCount = len(phaseLocationGuess) + 1
        super().__init__(bUseAdaptiveInterpolation=adaptiveInterpolation, returnValueCount=returnValueCount)

        self.effectivePotential = effectivePotential 
        self.phaseLocationGuess = phaseLocationGuess

    ## TEMP TEMP
    def __call__(self, x: npt.ArrayLike, useInterpolatedValues=True) -> npt.ArrayLike:
        #print(x)
        return super().__call__(x, useInterpolatedValues=useInterpolatedValues)

    def _functionImplementation(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """
        Parameters
        ----------
        temperature: float or numpy array of floats.
        """

        phaseLocation, potentialAtMinimum = self.effectivePotential.findLocalMinimum(self.phaseLocationGuess, temperature)

        # Important: Minimization may not always work as intended, 
        # for example the minimum we're looking for may not even exist at the input temperature.
        # This will break interpolations unless we validate the result here. 
        # InterpolatableFunction is constructed to ignore inputs where the function evaluated to np.nan, 
        # so we avoid issues by returning np.nan here if the minimization failed.

        # How to do the validation? Perhaps the safest way would be to call this in a T-loop and storing the phaseLocation
        # at the previous T. If phaseLocation is wildly different at the next T, this may suggest that we ended up in a different minimum.
        # Issue with this approach is that it doesn't vectorize. Might not be a big loss however since findLocalMinimum itself is 
        # not effectively vectorized due to reliance on scipy routines.

        # Here is a check that should catch "symmetry-breaking" type transitions where a field is 0 in one phase and nonzero in another
        bFieldWentToZero = (np.abs(self.phaseLocationGuess) > 5.0) & (np.abs(phaseLocation) < 1e-1)

        ## Check that we apply row-wise
        bEvaluationFailed = bFieldWentToZero ## & ... add other checks ...

        ## For scalar input let's return a 1D numpy array. Note ordering
        if (np.isscalar(temperature) or np.ndim(temperature) == 0):

            if (np.any(bEvaluationFailed)):
                return np.full(len(self.phaseLocationGuess) + 1, np.nan)

            res = np.asanyarray(phaseLocation)
            res = np.append(phaseLocation, potentialAtMinimum)

            return res
        
        else:
            ## Input was a numpy array, so output should be 2D. But if there's just 1 field it can be 1D.
            ## For consistency let's force it to be 2D.
            if (np.ndim(phaseLocation) == 1):
                phaseLocation = np.atleast_2d(phaseLocation)
                phaseLocation = phaseLocation.transpose()

            assert len(phaseLocation) == len(temperature)

            ## Make our failure check a boolean mask that numpy understands
            invalidRowMask = np.any(bEvaluationFailed, axis=1)

            ## Replace all elements with np.nan on rows that failed the check
            phaseLocation[invalidRowMask, :] = np.nan
            potentialAtMinimum[invalidRowMask] = np.nan
        

            # reshape so that potentialAtMinimum is a column vector
            potentialAtMinimum_column = potentialAtMinimum[:, np.newaxis]

            # Join the arrays so that potentialAtMinimum is the last column and the others are as in phaseLocation
            result = np.concatenate((phaseLocation, potentialAtMinimum_column), axis=1)
            return result


