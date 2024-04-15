from dataclasses import dataclass
import numpy as np
from .Fields import Fields
from .helpers import boostVelocity
# Put common data classes etc here


@dataclass
class PhaseInfo:
    # Field values at the two phases at T (we go from 1 to 2)
    phaseLocation1: Fields
    phaseLocation2: Fields
    temperature: float



"""LN: What's going on with the fieldProfiles array here? When constructing a background in EOM.wallPressure(), 
it explicitly reshapes the input fieldProfiles to include endpoints (VEVs). But then in this class there is a lot of slicing in range 1:-1
that just removes the endspoints.
"""
class BoltzmannBackground:
    def __init__(
        self,
        velocityMid: np.ndarray,
        velocityProfile: np.ndarray,
        fieldProfiles: Fields,
        temperatureProfile: np.ndarray,
        polynomialBasis: str = "Cardinal",
    ):
        # assumes input is in the wall frame
        self.vw = 0
        self.velocityProfile = np.asarray(velocityProfile)
        self.fieldProfiles = fieldProfiles.view(Fields) ## NEEDS to be Fields object
        self.temperatureProfile = np.asarray(temperatureProfile)
        self.polynomialBasis = polynomialBasis
        self.vMid = velocityMid
        self.TMid = 0.5 * (temperatureProfile[0] + temperatureProfile[-1])

    def boostToPlasmaFrame(self) -> None:
        """
        Boosts background to the plasma frame
        """
        self.velocityProfile = boostVelocity(self.velocityProfile, self.vMid)
        self.vw = boostVelocity(self.vw, self.vMid)

    def boostToWallFrame(self) -> None:
        """
        Boosts background to the wall frame
        """
        self.velocityProfile = boostVelocity(self.velocityProfile, self.vw)
        self.vw = 0


@dataclass
class BoltzmannDeltas:
    """
    Integrals of the out-of-equilibrium particle densities,
    defined in equation (15) of arXiv:2204.13120.
    """
    Delta00: np.ndarray
    Delta02: np.ndarray
    Delta20: np.ndarray
    Delta11: np.ndarray


@dataclass
class BoltzmannResults:
    """
    Holds results to be returned by BoltzmannSolver
    """
    deltaF: np.ndarray
    Deltas: BoltzmannDeltas
    truncationError: float
    
    # These two criteria are to evaluate the validity of the linearization of the 
    # Boltzmann equation. The tuples contain one element for each out-of-equilibrium
    # particle. To be valid, at least one criterion must be small for each particle.
    linearizationCriterion1: tuple
    linearizationCriterion2: tuple


@dataclass
class HydroResults():
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
class WallParams():
    ## Holds wall widths and wall offsets for all fields
    widths: np.ndarray ## 1D array
    offsets: np.ndarray ## 1D array

    def __add__(self, other):
        return WallParams(widths = (self.widths + other.widths), offsets = (self.offsets + other.offsets))

    def __sub__(self, other):
        return WallParams(widths = (self.widths - other.widths), offsets = (self.offsets - other.offsets))

    def __mul__(self, other):
        ## does not work if other = WallParams type
        return WallParams(widths = self.widths * other, offsets = self.offsets * other)

    def __truediv__(self, other):
        ## does not work if other = WallParams type
        return WallParams(widths = self.widths / other, offsets = self.offsets / other)


@dataclass
class WallGoResults:
    # HACK! This should probably go in its own file, Results.py,
    # but I was getting circular import errors, so hacked it here.
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
    Deltas: BoltzmannDeltas
    truncationError: float
    # finite difference results
    #deltaFFiniteDifference: np.ndarray
    #DeltasFiniteDifference: dict
    # measures of nonlinearity
    linearizationCriterion1: tuple
    linearizationCriterion2: tuple

    def __init__(self):
        pass

    def setWallVelocities(
        self,
        wallVelocity: float,
        wallVelocityError: float,
        wallVelocityLTE: float,
    ):
        # main results
        self.wallVelocity = wallVelocity
        self.wallVelocityError = wallVelocityError
        self.wallVelocityLTE = wallVelocityLTE

    def setHydroResults(self, hydroResults: HydroResults):
        # hydrodynamics results
        self.temperaturePlus = hydroResults.temperaturePlus
        self.temperatureMinus = hydroResults.temperatureMinus
        self.velocityJouget = hydroResults.velocityJouget

    def setWallParams(self, wallParams: WallParams):
        # quantities from WallParams
        self.wallWidths = wallParams.widths
        self.wallOffsets = wallParams.offsets

    def setBoltzmannBackground(self, boltzmannBackground: BoltzmannBackground):
        # quantities from BoltzmannBackground
        self.velocityProfile = boltzmannBackground.velocityProfile
        self.fieldProfiles = boltzmannBackground.fieldProfiles
        self.temperatureProfile = boltzmannBackground.temperatureProfile

    def setBoltzmannResults(self, boltzmannResults: BoltzmannResults):
        # quantities from BoltzmannResults
        self.deltaF = boltzmannResults.deltaF
        self.Deltas = boltzmannResults.Deltas
        self.linearizationCriterion1 = boltzmannResults.linearizationCriterion1
        self.linearizationCriterion2 = boltzmannResults.linearizationCriterion2
