import numpy as np
from dataclasses import dataclass

# WallGo imports
from .Fields import Fields
from .FreeEnergy import FreeEnergy
from .Polynomial import Polynomial
from .Thermodynamics import Thermodynamics


@dataclass
class WallGoResults:
    # bubble wall speed, and error
    wallVelocity: float
    wallVelocityError: float
    # local thermal equilibrium result
    wallVelocityLTE: float
    # hydrodynamic results
    velocityPlus: float
    velocityMinus: float
    velocityJouget: float
    # scalar fields
    fieldWidths: np.ndarray
    fieldOffsets: np.ndarray
    fieldProfile: Fields
    # background quantities
    temperatureProfile: np.ndarray
    velocityProfile: np.ndarray
    # deviations from equilibrium
    deltaFs: list[Polynomial]
    Deltas: list[dict]
    # finite difference results
    DeltasFiniteDifference: list[dict]
    # measures of nonlinearity
    nonlinearitys: np.ndarray

    def __init__(self):
        # HACK! This seems crazy - anyone have a better idea of how to
        # initialise and assign results?
        self.wallVelocity = None
        self.wallVelocityError = None
        self.wallVelocityLTE = None
        self.velocityPlus = None
        self.velocityMinus = None
        self.velocityJouget = None
        self.fieldWidths = None
        self.fieldOffsets = None
        self.fieldProfile = None
        self.temperatureProfile = None
        self.velocityProfile = None
        self.deltaFs = None
        self.Deltas = None
        self.DeltasFiniteDifference = None
        self.nonlinearities = None

