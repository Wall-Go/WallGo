import numpy as np
from abc import ABC, abstractmethod ## Abstract Base Class
import cmath # complex numbers

## WallGo imports
from WallSpeed import Particle


class GenericModel(ABC):
    '''
    Common interface for WallGo model definitions. 
    The user should implement this and the abstract methods below with their model-specific stuff. 
    '''

    ## LN: I don't know what the proper Python way of declaring interface variables is, but this works.
    ## In any case we certainly want to specify that any model should have these. 


    ## Model parameters (parameters in the action and RG scale, but not temperature) are expected to be a member dict.
    ## Here is a property definition for it. Child classes can just do modelParameters = { ... } to define it
    @property
    @abstractmethod
    def modelParameters(self) -> dict[str, float]:
        pass

    ## Particle array property, this should hold all particles 
    @property
    @abstractmethod
    def particles(self) -> np.ndarray[Particle]:
        pass


    ## Another particle array for holding just the out-of-equilibrium particles
    @property
    @abstractmethod
    def outOfEquilibriumParticles(self) -> np.ndarray[Particle]:
        pass
    

    ## Common routine for defining a new particle. Usually should not be overriden
    def addParticle(self, particleToAdd: Particle) -> None:
        self.particles = np.append(self.particles, particleToAdd)
        
        # add to out-of-eq particles too if applicable
        if (not particleToAdd.inEquilibrium):
            self.outOfEquilibriumParticles = np.append(self.outOfEquilibriumParticles, particleToAdd)



    @abstractmethod 
    def evaluateEffectivePotential(self, fields: np.ndarray[float], temperature: float) -> complex:
        ## This should calculate the potential given an array of background fields. 
        ## We allow the return value to be complex
        raise NotImplementedError

