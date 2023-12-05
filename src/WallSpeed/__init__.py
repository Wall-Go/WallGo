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

from .WallGoUtils import loadConfig
from .WallGoUtils import getPackagedDataPath

from .CollisionModuleLoader import loadCollisionModule, CollisionModule, collisionModuleLoaded

defaultConfigFile = getPackagedDataPath("WallSpeed.Config", "WallGoDefaults.ini")

config = loadConfig(defaultConfigFile)

if (config == {}):
    errorMessage = "Failed to load WallGo config file: " + defaultConfigFile
    raise RuntimeError(errorMessage)

print("Read WallGo config:")
print(config)

## Load the collision module, gets stored in CollisionModule global var
loadCollisionModule()