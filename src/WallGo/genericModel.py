from typing import Protocol
from abc import ABC, abstractmethod  # Abstract Base Class

## WallGo imports
from .particle import Particle
from .EffectivePotential import EffectivePotential


class GenericModel(ABC):
    """
    Common interface for WallGo model definitions.
    This is basically input parameters + particle definitions + effective potential.
    The user should implement this and the abstract methods below with their model-specific stuff.
    """

    def __init__(self):
        """Initializes empty model content.
        """
        self.particles: list[Particle] = []
        self.outOfEquilibriumParticles: list[Particle] = []
        self.modelParameters: dict[str, float] = {}
        effectivePotential: EffectivePotential = None
        inputParameters: dict[str, float] = None

        
    ## How many classical fields. We require getter only; field count should not change at runtime
    @property
    @abstractmethod
    def fieldCount(self) -> int:
        pass

    def addParticle(self, particleToAdd: Particle) -> None:
        ## Common routine for defining a new particle. Usually should not be overriden

        self.particles.append(particleToAdd)

        # add to out-of-eq particles too if applicable
        if not particleToAdd.inEquilibrium:
            self.outOfEquilibriumParticles.append(particleToAdd)

    ## Empties the particle lists
    def clearParticles(self) -> None:
        self.particles = []
        self.outOfEquilibriumParticles = []

    ## Go from whatever input parameters to renormalized Lagrangian parameters.
    # Override this if your inputs are something else than Lagrangian parameters
    def calculateModelParameters(
        self, inputParameters: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculates the model parameters based on the given input parameters.

        Args:
            inputParameters (dict[str, float]):
            A dictionary containing the input parameters.

        Returns:
            dict[str, float]:
            A dictionary containing the calculated model parameters.
        """
        self.inputParameters = inputParameters
        return {}
