import numpy as np
from dataclasses import dataclass

# WallGo imports
from .Boltzmann import BoltzmannBackground, BoltzmannResults
from .EOM import WallParams
from .Fields import Fields
from .Hydro import HydroResults


@dataclass
class WallGoResults:
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
    fieldProfile: Fields
    temperatureProfile: np.ndarray
    # quantities from BoltzmannResults
    deltaF: np.ndarray
    Deltas: dict
    truncationError: float
    # finite difference results
    #deltaFFiniteDifference: np.ndarray
    #DeltasFiniteDifference: dict
    # measures of nonlinearity
    #nonlinearitys: np.ndarray

    def __init__(
        self,
        wallVelocity: float,
        wallVelocityError: float,
        wallVelocityLTE: float,
        hydroResults: HydroResults,
        wallParams: WallParams,
        boltzmannBackground: BoltzmannBackground,
        boltzmannResults: BoltzmannResults
    ):
        # main results
        self.wallVelocity = wallVelocity
        self.wallVelocityError = wallVelocityError
        # hydrodynamics results
        self.wallVelocityLTE = wallVelocityLTE
        self.temperaturePlus = hydroResults.temperaturePlus
        self.temperatureMinus = hydroResults.temperatureMinus
        self.velocityJouget = hydroResults.velocityJouget
        # quantities from WallParams
        self.wallWidths = wallParams.widths
        self.wallOffsets = wallParams.offsets
        # quantities from BoltzmannBackground
        self.velocityProfile = boltzmannBackground.velocityProfile
        self.fieldProfile = boltzmannBackground.fieldProfile
        self.temperatureProfile = boltzmannBackground.temperatureProfile
        # quantities from BoltzmannResults
        self.deltaF = boltzmannResults.deltaF
        self.Deltas = boltzmannResults.Deltas
        self.truncationError = boltzmannResults.truncationError
        #self.deltaFFiniteDifference = ...
        #self.DeltasFiniteDifference = ...
        #self.nonlinearities = ...