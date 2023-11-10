from WallSpeed import EffectivePotential
from .InterpolatableFunction import InterpolatableFunction
#from WallSpeed import InterpolatableFunction

class FreeEnergy(InterpolatableFunction):
    def __init__(self, effectivePotential: EffectivePotential, initialGuess):
        super().__init__()
        self.effectivePotential = effectivePotential 
        self.initialGuess = initialGuess

    def _functionImplementation(self, xInput: float):
        """
        xInput: float or numpy array of floats.
        """
        def wrapper(x: float):
            res = lambda x: self.effectivePotential.findLocalMinimum(self.initialGuess , x)
            #res = lambda x: x**2
            # return res(x)
            return res(x)[1]
        
        return wrapper(xInput)