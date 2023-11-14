from .Boltzmann import BoltzmannBackground, BoltzmannSolver
from .Grid import Grid
from .Hydro import Hydro
from .HydroTemplateModel import HydroTemplateModel
from .model import FreeEnergy, Model, Particle
from .Polynomial import Polynomial
from .Thermodynamics import Thermodynamics
from .EOM import EOM

from .WallGoUtils import loadConfig
from .WallGoUtils import getProjectRoot

from .CollisionModuleLoader import loadCollisionModule, CollisionModule, collisionModuleLoaded

import os


defaultConfigFile = str(getProjectRoot()) + "/Config/WallGoDefaults.ini"
config = loadConfig(defaultConfigFile)

if (config == {}):
    errorMessage = "Failed to load WallGo config file: " + defaultConfigFile
    raise RuntimeError(errorMessage)

print("Read WallGo config:")
print(config)

## Load the collision module, gets stored in CollisionModule global var
loadCollisionModule()