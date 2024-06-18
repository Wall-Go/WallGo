"""
Data classes for compiling and returning results
"""

from dataclasses import dataclass
import numpy as np
from scipy.interpolate import UnivariateSpline
from .Fields import Fields
from .containers import BoltzmannBackground, BoltzmannDeltas


@dataclass
class BoltzmannResults:
    """
    Holds results to be returned by BoltzmannSolver
    """

    deltaF: np.ndarray
    Deltas: BoltzmannDeltas  # pylint: disable=invalid-name
    truncationError: float

    # These two criteria are to evaluate the validity of the linearization of the
    # Boltzmann equation. The arrays contain one element for each out-of-equilibrium
    # particle. To be valid, at least one criterion must be small for each particle.
    linearizationCriterion1: np.ndarray
    linearizationCriterion2: np.ndarray

    def __mul__(self, number: float) -> "BoltzmannResults":
        return BoltzmannResults(
            deltaF=number * self.deltaF,
            Deltas=number * self.Deltas,
            truncationError=abs(number) * self.truncationError,
            linearizationCriterion1=abs(number) * self.linearizationCriterion1,
            linearizationCriterion2=self.linearizationCriterion2,
        )

    def __rmul__(self, number: float) -> "BoltzmannResults":
        return BoltzmannResults(
            deltaF=number * self.deltaF,
            Deltas=number * self.Deltas,
            truncationError=abs(number) * self.truncationError,
            linearizationCriterion1=abs(number) * self.linearizationCriterion1,
            linearizationCriterion2=self.linearizationCriterion2,
        )

    def __add__(self, other: "BoltzmannResults") -> "BoltzmannResults":
        return BoltzmannResults(
            deltaF=other.deltaF + self.deltaF,
            Deltas=other.Deltas + self.Deltas,
            truncationError=other.truncationError + self.truncationError,
            linearizationCriterion1=other.linearizationCriterion1
            + self.linearizationCriterion1,
            linearizationCriterion2=other.linearizationCriterion2
            + self.linearizationCriterion2,
        )

    def __sub__(self, other: "BoltzmannResults") -> "BoltzmannResults":
        return self.__add__((-1) * other)


@dataclass
class HydroResults:
    """
    Holds results to be returned by Hydro
    """

    # hydrodynamic results
    temperaturePlus: float
    temperatureMinus: float
    velocityJouget: float

    def __init__(
        self,
        temperaturePlus: float,
        temperatureMinus: float,
        velocityJouget: float,
    ):
        self.temperaturePlus = temperaturePlus
        self.temperatureMinus = temperatureMinus
        self.velocityJouget = velocityJouget


@dataclass
class WallParams:
    """
    Holds wall widths and wall offsets for all fields
    """
    widths: np.ndarray  ## 1D array
    offsets: np.ndarray  ## 1D array

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


@dataclass
class WallGoResults:
    """
    Compiles output results for users of WallGo
    """
    # bubble wall speed, and error
    wallVelocity: float
    wallVelocityError: float
    # local thermal equilibrium result
    wallVelocityLTE: float
    # hydrodynamic results
    temperaturePlus: float
    temperatureMinus: float
    velocityJouget: float
    # quantities from WallParams
    wallWidths: np.ndarray
    wallOffsets: np.ndarray
    # quantities from BoltzmannBackground
    velocityProfile: np.ndarray
    fieldProfiles: Fields
    temperatureProfile: np.ndarray
    # quantities from BoltzmannResults
    deltaF: np.ndarray
    Deltas: BoltzmannDeltas  # pylint: disable=invalid-name
    truncationError: float
    # finite difference results
    deltaFFiniteDifference: np.ndarray
    DeltasFiniteDifference: BoltzmannDeltas  # pylint: disable=invalid-name
    # measures of nonlinearity
    linearizationCriterion1: np.ndarray
    linearizationCriterion2: np.ndarray

    def __init__(self) -> None:
        pass

    def setWallVelocities(
        self,
        wallVelocity: float,
        wallVelocityError: float,
        wallVelocityLTE: float,
    ) -> None:
        """
        Set wall velocity results
        """
        self.wallVelocity = wallVelocity
        self.wallVelocityError = wallVelocityError
        self.wallVelocityLTE = wallVelocityLTE

    def setHydroResults(self, hydroResults: HydroResults) -> None:
        """
        Set hydrodynamics results
        """
        self.temperaturePlus = hydroResults.temperaturePlus
        self.temperatureMinus = hydroResults.temperatureMinus
        self.velocityJouget = hydroResults.velocityJouget

    def setWallParams(self, wallParams: WallParams) -> None:
        """
        Set wall parameters results
        """
        self.wallWidths = wallParams.widths
        self.wallOffsets = wallParams.offsets

    def setBoltzmannBackground(self, boltzmannBackground: BoltzmannBackground) -> None:
        """
        Set Boltzmann background results
        """
        self.velocityProfile = boltzmannBackground.velocityProfile
        self.fieldProfiles = boltzmannBackground.fieldProfiles
        self.temperatureProfile = boltzmannBackground.temperatureProfile

    def setBoltzmannResults(self, boltzmannResults: BoltzmannResults) -> None:
        """
        Set Boltzmann results
        """
        self.deltaF = boltzmannResults.deltaF
        self.Deltas = boltzmannResults.Deltas
        self.truncationError = boltzmannResults.truncationError
        self.linearizationCriterion1 = boltzmannResults.linearizationCriterion1
        self.linearizationCriterion2 = boltzmannResults.linearizationCriterion2

    def setFiniteDifferenceBoltzmannResults(
        self, boltzmannResults: BoltzmannResults
    ) -> None:
        """
        Set finite difference Boltzmann results
        """
        self.deltaFFiniteDifference = boltzmannResults.deltaF
        self.DeltasFiniteDifference = boltzmannResults.Deltas


@dataclass
class WallGoInterpolationResults:
    """
    Used when interpolating the pressure. Like WallGoResults but expanded to lists.
    """
    ## List of stable solutions
    wallVelocities: list[float]
    ## List of unstable solutions
    unstableWallVelocities: list[float]

    ## Velocity grid on which the pressures were computed
    velocityGrid: list[float]
    ## Pressures evaluated at velocityGrid
    pressures: list[float]
    ## Spline of the pressure
    pressureSpline: UnivariateSpline

    ## WallParams evaluated at velocityGrid
    wallParams: list[WallParams]
    ## BoltzmannResults evaluated at velocityGrid
    boltzmannResults: list[BoltzmannResults]
    ## BoltzmannBackground evaluated at velocityGrid
    boltzmannBackgrounds: list[BoltzmannBackground]
    ## HydroResults evaluated at velocityGrid
    hydroResults: list[HydroResults]
