"""
Data classes passed around WallGo
"""
from dataclasses import dataclass
import numpy as np
from .fields import Fields
from .helpers import boostVelocity
from .polynomial import Polynomial


@dataclass
class PhaseInfo:
    """ Object describing coexisting phases.
    """

    phaseLocation1: Fields
    """Field value of the starting phase."""

    phaseLocation2: Fields
    """Field value of the ending phase."""

    temperature: float
    """Temperature of transition."""


@dataclass
class WallParams:
    """
    Holds wall widths and wall offsets for all fields
    """

    widths: np.ndarray  # 1D array
    """Bubble wall widths in each field direction. Should be expressed in physical units
    (the units used in EffectivePotential)."""

    offsets: np.ndarray  # 1D array
    """Bubble wall offsets in each field direction."""

    def __add__(self, other: "WallParams") -> "WallParams":
        return WallParams(
            widths=(self.widths + other.widths), offsets=(self.offsets + other.offsets)
        )

    def __sub__(self, other: "WallParams") -> "WallParams":
        return WallParams(
            widths=(self.widths - other.widths), offsets=(self.offsets - other.offsets)
        )

    def __mul__(self, number: float) -> "WallParams":
        ## does not work if other = WallParams type
        return WallParams(widths=self.widths * number, offsets=self.offsets * number)

    def __rmul__(self, number: float) -> "WallParams":
        return self.__mul__(number)

    def __truediv__(self, number: float) -> "WallParams":
        ## does not work if other = WallParams type
        return WallParams(widths=self.widths / number, offsets=self.offsets / number)


class BoltzmannBackground:
    """
    Container for holding velocity, temperature and field backgrounds on which
    out-of-equilibrium fluctuations live.
    """

    velocityWall: float
    """Bubble wall velocity."""

    velocityMid: float
    """The average between the asymptotic velocities inside and outside the bubble."""

    velocityProfile: np.ndarray
    """Fluid velocity as a function of position."""

    fieldProfiles: Fields
    """Field profile as a function of position."""

    temperatureProfile: np.ndarray
    """Temperarture profile as a function of position."""

    polynomialBasis: str
    """Type of polynomial basis used, e.g. Chebyshev, Cardinal."""

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
    r"""Relativistically invariant integral over :math:`\delta f`."""

    Delta02: Polynomial  # pylint: disable=invalid-name
    r"""Relativistically invariant integral over
    :math:`\mathcal{P}^2_\text{pl}\delta f`."""

    Delta20: Polynomial  # pylint: disable=invalid-name
    r"""Relativistically invariant integral over
    :math:`\mathcal{E}^2_\text{pl}\delta f`."""

    Delta11: Polynomial  # pylint: disable=invalid-name
    r"""Relativistically invariant integral over
    :math:`\mathcal{E}_\text{pl}\mathcal{P}_\text{pl}\delta f`."""

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
