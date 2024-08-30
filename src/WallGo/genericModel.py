"""Physics model class for WallGo"""

from abc import ABC, abstractmethod  # Abstract Base Class
from typing import Optional

## WallGo imports
from .particle import Particle
from .EffectivePotential import EffectivePotential


class GenericModel(ABC):
    """
    Common interface for WallGo model definitions.
    This is basically input parameters + particle definitions + effective potential.
    The user should implement this and the abstract methods below
    with their model-specific stuff.
    """

    def __init__(self) -> None:
        """Initializes empty model content."""
        self.particles: list[Particle] = []
        self.outOfEquilibriumParticles: list[Particle] = []
        self.modelParameters: dict[str, float] = {}
        self.effectivePotential: Optional[EffectivePotential] = None
        self.inputParameters: dict[str, float] = {}

    @property
    @abstractmethod
    def fieldCount(self) -> int:
        """Override to return the number of classical background fields
        in your model."""

    def addParticle(self, particleToAdd: Particle) -> None:
        """Common routine for defining a new particle."""

        self.particles.append(particleToAdd)

        # add to out-of-eq particles too if applicable
        if not particleToAdd.inEquilibrium:
            self.outOfEquilibriumParticles.append(particleToAdd)

    def clearParticles(self) -> None:
        """Empties internal particle lists"""
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
