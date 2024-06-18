from dataclasses import dataclass
import numpy as np
from .Fields import Fields
from .helpers import boostVelocity
from .Polynomial import Polynomial
from scipy.interpolate import UnivariateSpline
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
