"""Import types here. We do this so that eg. the EOM class can be accessed as 
WallGo.EOM. If this wasn't done, WallGo.EOM would actually refer to the MODULE EOM.py which we don't want,
and would cause hard-to-diagnoze crashes.
TODO Is there a better way of doing all this?! 
"""

from .Boltzmann import BoltzmannBackground, BoltzmannSolver
from .exceptions import WallGoError, WallGoPhaseValidationError
from .Grid import Grid
from .Hydro import Hydro
from .HydroTemplateModel import HydroTemplateModel
from .Polynomial import Polynomial
from .Thermodynamics import Thermodynamics
from .EOM import EOM, WallGoResults
from .EOM import WallParams
from .WallGoTypes import PhaseInfo


from .Particle import Particle
from .Fields import Fields
from .GenericModel import GenericModel
from .EffectivePotential import EffectivePotential
from .EffectivePotential_NoResum import EffectivePotential_NoResum
from .FreeEnergy import FreeEnergy
from .WallGoManager import WallGoManager
from .WallGoManager import PhaseInfo
from .InterpolatableFunction import InterpolatableFunction

from .CollisionArray import CollisionArray

from .Integrals import Integrals
from .Config import Config

from .collision_wrapper import Collision
from .WallGoUtils import getSafePathToResource


defaultConfigFile = getSafePathToResource("Config/WallGoDefaults.ini")

#config = loadConfig(defaultConfigFile)

#if (config == {}):
#    errorMessage = "Failed to load WallGo config file: " + defaultConfigFile
#    raise RuntimeError(errorMessage)

#print("Read WallGo config:")
#print(config)

_bInitialized = False
config = Config()

"""Default integral objects for WallGo. Calling WallGo.initialize() optimizes these by replacing their direct computation with 
precomputed interpolation tables."""
defaultIntegrals = Integrals()
defaultIntegrals.Jb.disableAdaptiveInterpolation()
defaultIntegrals.Jf.disableAdaptiveInterpolation()


## Define a separate initializer function that does NOT get called automatically. 
## This is good for preventing heavy startup operations from running if the user just wants a one part of WallGo and not the full framework, eg. ``import WallGo.Integrals``.
## Downside is that programs need to manually call this, preferably as early as possible.
def initialize() -> None:
    """WallGo initializer. This should be called as early as possible in your program.
    """

    global _bInitialized
    global config 

    if not _bInitialized:

        ## read default config
        config.readINI( getSafePathToResource("Config/WallGoDefaults.ini") )
        
        #print(config)

        ## Initialize interpolations for our default integrals
        _initalizeIntegralInterpolations()

        _bInitialized = True
    
    else:
        raise RuntimeWarning("Warning: Repeated call to WallGo.initialize()")
    

def _initalizeIntegralInterpolations() -> None:
    global config 
    
    defaultIntegrals.Jb.readInterpolationTable(
        getSafePathToResource(config.get("DataFiles", "InterpolationTable_Jb")), bVerbose=False 
        )
    defaultIntegrals.Jf.readInterpolationTable(
        getSafePathToResource(config.get("DataFiles", "InterpolationTable_Jf")), bVerbose=False 
        )