"""Physics model class for WallGo"""

from abc import ABC, abstractmethod  # Abstract Base Class
from typing import Type, Any

## WallGo imports
from .particle import Particle
from .effectivePotential import EffectivePotential


class GenericModel(ABC):
    """
    Common interface for WallGo model definitions.
    This is basically input parameters + particle definitions + effective potential.
    The user should implement this and the abstract methods below
    with their model-specific stuff.
    """

    @property
    @abstractmethod
    def fieldCount(self) -> int:
        """Override to return the number of classical background fields
        in your model."""

    @abstractmethod
    def getEffectivePotential(self) -> "EffectivePotential":
        """Override to return your effective potential."""

    def __init_subclass__(cls: Type["GenericModel"], **kwargs: Any) -> None:
        """Called whenever a subclass is initialized.
        Initialize particle list here.
        """
        super().__init_subclass__(**kwargs)
        cls.outOfEquilibriumParticles = []

    ######

    def addParticle(self, particleToAdd: Particle) -> None:
        """Common routine for defining a new out-of-equilibrium particle."""
        self.outOfEquilibriumParticles.append(particleToAdd)

    def clearParticles(self) -> None:
        """Empties the cached particle list"""
        self.outOfEquilibriumParticles: list[Particle] = []
