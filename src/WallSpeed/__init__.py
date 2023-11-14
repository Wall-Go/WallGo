from .Boltzmann import BoltzmannBackground, BoltzmannSolver
from .Grid import Grid
from .Hydro import Hydro
from .HydroTemplateModel import HydroTemplateModel
from .model import FreeEnergy, Model, Particle
from .Polynomial import Polynomial
from .Thermodynamics import Thermodynamics
from .EOM import EOM

from .LoadConfig import loadConfig

from .CollisionModuleLoader import loadCollisionModule, CollisionModule, collisionModuleLoaded

import os


WallGoRootDir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

defaultConfigFile = WallGoRootDir + "/Config/WallGoDefaults.ini"
config = loadConfig(defaultConfigFile)

if (config == {}):
    raise RuntimeError(f"Failed to load WallGo config file: {defaultConfigFile}")

print("Read WallGo config:")
print(config)

## Load the collision module, gets stored in CollisionModule global var
loadCollisionModule()