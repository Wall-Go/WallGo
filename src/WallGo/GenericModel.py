import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod ## Abstract Base Class

## WallGo imports
from .Particle import Particle
from .EffectivePotential import EffectivePotential



class GenericModel(ABC):
    '''
    Common interface for WallGo model definitions. This is basically input parameters + particle definitions + effective potential.
    The user should implement this and the abstract methods below with their model-specific stuff. 
    '''


    ## Particle list, this should hold all particles relevant for matrix elements (including in-equilibrium ones) 
    @property
    @abstractmethod
    def particles(self) -> list[Particle]:
        pass


    ## Another particle array for holding just the out-of-equilibrium particles
    @property
    @abstractmethod
    def outOfEquilibriumParticles(self) -> list[Particle]:
        pass

    ## Model parameters (parameters in the action and RG scale, but not temperature) are expected to be a member dict.
    ## Here is a property definition for it. Child classes can just do modelParameters = { ... } to define it
    @property
    @abstractmethod
    def modelParameters(self) -> dict[str, float]:
        pass

    ## How many classical fields
    @property
    @abstractmethod
    def fieldCount(self) -> int:
        pass


    '''
    ## Effective potential
    @property
    @abstractmethod
    def Veff(self) -> EffectivePotential:
        pass
    '''
    effectivePotential: EffectivePotential
    
    inputParameters: dict[str, float]
    


    def addParticle(self, particleToAdd: Particle) -> None:
        ## Common routine for defining a new particle. Usually should not be overriden

        self.particles.append(particleToAdd)
        
        # add to out-of-eq particles too if applicable
        if (not particleToAdd.inEquilibrium):
            self.outOfEquilibriumParticles.append(particleToAdd)


    ## Go from whatever input parameters to renormalized Lagrangian parameters. Override this if your inputs are something else than Lagrangian parameters
    def calculateModelParameters(self, inputParameters: dict[str, float]) -> dict[str, float]:
        self.inputParameters = inputParameters
        return {}

    ## Redefine the modelParameters and effectivePotential based on new inputparameters
    def changeInputParameters(self, inputParameters: dict[str, float], effectivePotential: EffectivePotential) -> None:
        self.modelParameters = self.calculateModelParameters(inputParameters)
        self.effectivePotential = effectivePotential(self.modelParameters, self.fieldCount)

