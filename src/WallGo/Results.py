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
    vw: float
    vwErr: float
    # thermodynamics
    thermodynamics: Thermodynamics
    # hydrodynamic results
    vp: float
    vm: float
    vJ: float
    # background quantities
    fieldProfile: Fields
    temperatureProfile: np.ndarray
    velocityProfile: np.ndarray
    # deviations from equilibrium
    deltaFs: list[Polynomial]
    Deltas: list[dict]
    # finite difference results
    DeltasFiniteDifference: list[dict]
    # measures of nonlinearity
    nonlinearitys: np.ndarray

