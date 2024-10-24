"""Initialising WallGo package"""

import warnings

# subpackage
from . import PotentialTools

# package level modules
from .boltzmann import BoltzmannSolver
from .collisionArray import CollisionArray
from .containers import PhaseInfo, BoltzmannBackground, BoltzmannDeltas, WallParams
from .effectivePotential import EffectivePotential, VeffDerivativeSettings
from .exceptions import WallGoError, WallGoPhaseValidationError, CollisionLoadError
from .fields import Fields
from .freeEnergy import FreeEnergy
from .genericModel import GenericModel
from .grid import Grid
from .grid3Scales import Grid3Scales
from .hydrodynamics import Hydrodynamics
from .hydrodynamicsTemplateModel import HydrodynamicsTemplateModel
from .interpolatableFunction import InterpolatableFunction, EExtrapolationType
from .manager import WallGoManager, WallSolverSettings
from .particle import Particle
from .polynomial import Polynomial
from .thermodynamics import Thermodynamics
from .equationOfMotion import EOM
from .results import WallGoResults
from .utils import getSafePathToResource

from .config import Config


global _bCollisionModuleAvailable  # pylint: disable=invalid-name
_bCollisionModuleAvailable: bool = False

try:
    import WallGoCollision

    #print(f"Loaded WallGoCollision package from location: {WallGoCollision.__path__}")
    _bCollisionModuleAvailable = True  # pylint: disable=invalid-name

    from .collisionHelpers import *

except ImportError as e:
    warnings.warn(f"Error loading WallGoCollision module: {e}"
        "This could indicate an issue with your installation of WallGo or "
        "WallGoCollision, or both. This is non-fatal, but you will not be able to"
        " utilize collision integration routines."
    )


def isCollisionModuleAvailable() -> bool:
    """
    Returns True if the WallGoCollision extension module could be loaded and is ready
    for use. By default it is loaded together with WallGo, but WallGo can operate in
    restricted mode even if the load fails. This function can be used to check module
    availability at runtime if you must operate in an environment where the module may
    not always be available.
    """
    return _bCollisionModuleAvailable

_bInitialized = False  # pylint: disable=invalid-name

# Define a separate initializer function that does NOT get called automatically.
# This is good for preventing heavy startup operations from running if the user just
# wants a one part of WallGo and not the full framework, eg. `import WallGo.Integrals`.
# Downside is that programs need to manually call this, preferably as early as possible.
def _initializeInternal() -> None:
    """
    WallGo initializer. This should be called as early as possible in your program.
    """

    global _bInitialized  # pylint: disable=invalid-name

    if not _bInitialized:
        _bInitialized = True
    else:
        raise RuntimeWarning("Warning: Repeated call to WallGo._initializeInternal()")
