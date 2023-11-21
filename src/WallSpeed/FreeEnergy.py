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
        returnValueCount = len(phaseLocationGuess) + 1
        super().__init__(bUseAdaptiveInterpolation=adaptiveInterpolation, returnValueCount=returnValueCount)

        self.effectivePotential = effectivePotential 
        self.phaseLocationGuess = phaseLocationGuess


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
        fieldWentToZero = (np.abs(self.phaseLocationGuess) > 1.0) & (np.abs(phaseLocation) < 1e-4)
        if (np.any(fieldWentToZero)):
            return np.nan ### TODO correct shape!!!!!!!!!!!!!!!!
        
        
        if (np.isscalar(temperature)):
            return phaseLocation, potentialAtMinimum
        
        else:
            # reshape so that potentialAtMinimum is a column vector
            potentialAtMinimum_column = potentialAtMinimum[:, np.newaxis]

            # Join the arrays so that potentialAtMinimum is the first column and the others are as in phaseLocation
            result = np.concatenate((potentialAtMinimum_column, phaseLocation), axis=1)
            return result


