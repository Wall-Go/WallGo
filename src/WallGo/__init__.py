"""Import types here. We do this so that eg. the EOM class can be accessed as 
WallGo.EOM. If this wasn't done, WallGo.EOM would actually refer to the MODULE EOM.py
which we don't want, and would cause hard-to-diagnoze crashes.
TODO Is there a better way of doing all this?! 
"""

from .boltzmann import BoltzmannSolver
from .containers import PhaseInfo, BoltzmannBackground, BoltzmannDeltas, WallParams
from .exceptions import WallGoError, WallGoPhaseValidationError
from .grid import Grid
from .grid3Scales import Grid3Scales
from .hydrodynamics import Hydrodynamics
from .hydrodynamicsTemplateModel import HydrodynamicsTemplateModel
from .polynomial import Polynomial
from .thermodynamics import Thermodynamics
from .equationOfMotion import EOM
from .results import WallGoResults


from .particle import Particle
from .fields import Fields
from .genericModel import GenericModel
from .EffectivePotential import EffectivePotential
from .FreeEnergy import FreeEnergy
from .wallGoManager import WallGoManager
from .InterpolatableFunction import InterpolatableFunction

from .collisionArray import CollisionArray

from .Integrals import Integrals
from .Config import Config

from .WallGoUtils import getSafePathToResource

global _bCollisionModuleAvailable # pylint: disable=invalid-name
_bCollisionModuleAvailable: bool = False

try:
    import WallGoCollision

    print(f"Loaded WallGoCollision package from location: {WallGoCollision.__path__}")
    _bCollisionModuleAvailable = True # pylint: disable=invalid-name

    from .collisionHelpers import *

except ModuleNotFoundError as e:
    print(f"Error loading WallGoCollision module: {e}")
    print(
        "This could indicate an issue with your installation of WallGo or "\
        "WallGoCollision, or both. This is non-fatal, but you will not be able to"\
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


defaultConfigFile = getSafePathToResource("Config/WallGoDefaults.ini")

# config = loadConfig(defaultConfigFile)

# if (config == {}):
#    errorMessage = "Failed to load WallGo config file: " + defaultConfigFile
#    raise RuntimeError(errorMessage)

# print("Read WallGo config:")
# print(config)

_bInitialized = False # pylint: disable=invalid-name
"""Configuration settings for WallGo"""
config = Config()

"""Default integral objects for WallGo. Calling WallGo.initialize() optimizes these by
replacing their direct computation with precomputed interpolation tables."""
defaultIntegrals = Integrals()
defaultIntegrals.Jb.disableAdaptiveInterpolation()
defaultIntegrals.Jf.disableAdaptiveInterpolation()


# Define a separate initializer function that does NOT get called automatically.
# This is good for preventing heavy startup operations from running if the user just
# wants a one part of WallGo and not the full framework, eg. `import WallGo.Integrals`.
# Downside is that programs need to manually call this, preferably as early as possible.
def initialize() -> None:
    """
    WallGo initializer. This should be called as early as possible in your program.
    """

    global _bInitialized # pylint: disable=invalid-name
    global config # pylint: disable=invalid-name

    if not _bInitialized:

        ## read default config
        config.readINI(getSafePathToResource("Config/WallGoDefaults.ini"))
        config.readINI(getSafePathToResource("Config/CollisionDefaults.ini"))

        # print(config)

        ## Initialize interpolations for our default integrals
        _initalizeIntegralInterpolations()

        _bInitialized = True

    else:
        raise RuntimeWarning("Warning: Repeated call to WallGo.initialize()")


def _initalizeIntegralInterpolations() -> None: # pylint: disable=invalid-name
    global config # pylint: disable=invalid-name

    defaultIntegrals.Jb.readInterpolationTable(
        getSafePathToResource(config.get("DataFiles", "InterpolationTable_Jb")),
        bVerbose=False,
    )
    defaultIntegrals.Jf.readInterpolationTable(
        getSafePathToResource(config.get("DataFiles", "InterpolationTable_Jf")),
        bVerbose=False,
    )
