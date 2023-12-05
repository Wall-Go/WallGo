from .Boltzmann import BoltzmannBackground, BoltzmannSolver
from .Grid import Grid
from .Hydro import Hydro
from .HydroTemplateModel import HydroTemplateModel
from .model import Model
from .Polynomial import Polynomial
from .Thermodynamics import Thermodynamics
from .EOM import EOM

from .Particle import Particle
from .GenericModel import GenericModel
from .EffectivePotential import EffectivePotential
from .FreeEnergy import FreeEnergy 
from .WallGoManager import WallGoManager
from .InterpolatableFunction import InterpolatableFunction

## Import Integrals.py explicitly and alias it because I like namespaces
from . import Integrals as Integrals

from .WallGoUtils import getSafePathToResource

from .CollisionModuleLoader import loadCollisionModule, CollisionModule, collisionModuleLoaded

defaultConfigFile = getSafePathToResource("Config/WallGoDefaults.ini")

#config = loadConfig(defaultConfigFile)

#if (config == {}):
#    errorMessage = "Failed to load WallGo config file: " + defaultConfigFile
#    raise RuntimeError(errorMessage)

#print("Read WallGo config:")
#print(config)

## Load the collision module, gets stored in CollisionModule global var
loadCollisionModule()

## Flag for checking if initialize() has been ran
_bInitialized = False

## Define a separate initializer function that does NOT get called automatically. 
## This is good for preventing heavy startup operations from running if the user just wants a one part of WallGo and not the full framework, eg. ``import WallGo.Integrals``.
## Downside is that programs need to manually call this, preferably as early as possible.
def initialize() -> None:
    """WallGo initializer. This should be called as early as possible in your program."""

    global _bInitialized
    if not _bInitialized:

        ## read config
        # ...

        _initalizeIntegralInterpolations()

        _bInitialized = True
    
    else:
        raise RuntimeWarning("Warning: Repeated call to WallGo.initialize()")
    

def _initalizeIntegralInterpolations() -> None:
    ## TODO read these paths from config
    Integrals.Jb.readInterpolationTable(getSafePathToResource("Data/InterpolationTable_Jb.txt"))
    Integrals.Jf.readInterpolationTable(getSafePathToResource("Data/InterpolationTable_Jf.txt"))



