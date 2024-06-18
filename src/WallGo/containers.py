"""
Data classes passed around WallGo
"""
from dataclasses import dataclass
import numpy as np
from .Fields import Fields
from .helpers import boostVelocity
from .Polynomial import Polynomial


@dataclass
class PhaseInfo:
    """
    Field values at the two phases at a given temperature (we go from 1 to 2)
    """

    phaseLocation1: Fields
    phaseLocation2: Fields
    temperature: float


class BoltzmannBackground:
    """
    Container for holding velocity, temperature and field backgrounds on which
    out-of-equilibrium fluctuations live.
    """
    velocityWall: float
    velocityMid: float
    velocityProfile: np.ndarray
    fieldProfiles: Fields
    temperatureProfile: np.ndarray
    polynomialBasis: str

    def __init__(
        self,
        velocityMid: float,
        velocityProfile: np.ndarray,
        fieldProfiles: Fields,
        temperatureProfile: np.ndarray,
        polynomialBasis: str = "Cardinal",
    ):
        # assumes input is in the wall frame
        self.velocityWall = 0
        self.velocityProfile = np.asarray(velocityProfile)
        self.fieldProfiles = fieldProfiles.view(Fields)  ## NEEDS to be Fields object
        self.temperatureProfile = np.asarray(temperatureProfile)
        self.polynomialBasis = polynomialBasis
        self.velocityMid = velocityMid

    def boostToPlasmaFrame(self) -> None:
        """
        Boosts background to the plasma frame
        """
        self.velocityProfile = boostVelocity(self.velocityProfile, self.velocityMid)
        self.velocityWall = boostVelocity(self.velocityWall, self.velocityMid)

    def boostToWallFrame(self) -> None:
        """
        Boosts background to the wall frame
        """
        self.velocityProfile = boostVelocity(self.velocityProfile, self.velocityWall)
        self.velocityWall = 0


@dataclass
class BoltzmannDeltas:
    """
    Integrals of the out-of-equilibrium particle densities,
    defined in equation (15) of arXiv:2204.13120.
    """

    Delta00: Polynomial  # pylint: disable=invalid-name
    Delta02: Polynomial  # pylint: disable=invalid-name
    Delta20: Polynomial  # pylint: disable=invalid-name
    Delta11: Polynomial  # pylint: disable=invalid-name

    # string literal type hints as class not defined yet
    def __mul__(self, number: float) -> "BoltzmannDeltas":
        """
        Multiply a BoltzmannDeltas object with a scalar.
        """
        return BoltzmannDeltas(
            Delta00=number * self.Delta00,
            Delta02=number * self.Delta02,
            Delta20=number * self.Delta20,
            Delta11=number * self.Delta11,
        )

    def __rmul__(self, number: float) -> "BoltzmannDeltas":
        """
        Multiply a BoltzmannDeltas object with a scalar.
        """
        return BoltzmannDeltas(
            Delta00=number * self.Delta00,
            Delta02=number * self.Delta02,
            Delta20=number * self.Delta20,
            Delta11=number * self.Delta11,
        )

    def __add__(self, other: "BoltzmannDeltas") -> "BoltzmannDeltas":
        """
        Add two BoltzmannDeltas objects.
        """
        return BoltzmannDeltas(
            Delta00=other.Delta00 + self.Delta00,
            Delta02=other.Delta02 + self.Delta02,
            Delta20=other.Delta20 + self.Delta20,
            Delta11=other.Delta11 + self.Delta11,
        )

    def __sub__(self, other: "BoltzmannDeltas") -> "BoltzmannDeltas":
        """
        Substract two BoltzmannDeltas objects.
        """
        return self.__add__((-1) * other)
