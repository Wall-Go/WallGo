import numpy as np
import numpy.typing as npt

from .InterpolatableFunction import InterpolatableFunction
from .EffectivePotential import EffectivePotential

class FreeEnergy(InterpolatableFunction):

    def __init__(self, effectivePotential: EffectivePotential, phaseLocationGuess):
        super().__init__()
        # super().__init__(bUseAdaptiveInterpolation=False)
        self.effectivePotential = effectivePotential 
        self.phaseLocationGuess = phaseLocationGuess

    # class fieldAtMinimum(InterpolatableFunction):
    #     def _functionImplementation(self, fieldValue: npt.ArrayLike) -> npt.ArrayLike:
    #         return fieldValue


    def _functionImplementation(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """
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
        # Issue with this approach is that it doesn't vectorize

        # Here is a check that should catch "symmetry-breaking" type transitions where a field is 0 in one phase and nonzero in another
        fieldWentToZero = (np.abs(self.phaseLocationGuess) > 1.0) & (np.abs(phaseLocation) < 1e-4)
        if (np.any(fieldWentToZero)):
            return np.nan

        return potentialAtMinimum




