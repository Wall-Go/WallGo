"""Simple example file for using the WallGo collision module from Python.
The module is written in C++ but implements Python bindings. There is also a pure Python wrapper class,
WallGo.Collision, which handles dynamic loading of the module and provides complementary functionality.
Note that WallGo.Collision is a singleton class, ie. only one instance of it can exist.
Loading of the module happens when the instance is first created."""

import os
import pathlib

import WallGo
from WallGo import Particle
from WallGo import Fields

from SingletStandardModel_Z2 import (
    SingletSM_Z2,
)  # Benoit benchmark model


## TODO move this to the collision wrapper:

## Convert Python 'Particle' object to pybind-bound ParticleSpecies object.
## But 'Particle' uses masses in GeV^2 units while we need m^2/T^2, so T is needed as input here.
## Should do the same for field values since the vacuum mass can depend on that.
## Return value is a ParticleSpecies object
def constructPybindParticle(particle: Particle, T: float, fields: Fields):
    r"""
        Converts 'Particle' object to ParticleSpecies object that the Collision module can understand.
        CollisionModule operates with dimensionless (m/T)^2 etc, so the temperature is taken as an input here. 

        Parameters
        ----------
        particle : Particle
            Particle object with p.msqVacuum and p.msqThermal being in GeV^2 units.
        T : float
            Temperature in GeV units.

        Returns
        -------
        CollisionModule.ParticleSpecies
            ParticleSpecies object
    """


    ## Convert to correct enum for particle statistics
    particleType = None
    if particle.statistics == "Boson":
        particleType = WallGo.Collision().module.EParticleType.BOSON
    elif particle.statistics == "Fermion":
        particleType =  WallGo.Collision().module.EParticleType.FERMION

    ## Hack vacuum masses are ignored
    return WallGo.Collision().module.ParticleSpecies(particle.name, particleType, particle.inEquilibrium, 
                                particle.msqVacuum(fields) / T**2.0, particle.msqThermal(T) / T**2.0,  particle.ultrarelativistic)


WallGo.initialize()


## Modify the config, we use N=5 for this example
WallGo.config.config.set("PolynomialGrid", "momentumGridSize", "5")

## QFT model input. Some of these are probably not intended to change, like gauge masses. Could hardcode those directly in the class.
inputParameters = {
    #"RGScale" : 91.1876,
    "RGScale" : 125., # <- Benoit benchmark
    "v0" : 246.0,
    "MW" : 80.379,
    "MZ" : 91.1876,
    "Mt" : 173.0,
    "g3" : 1.2279920495357861,
    # scalar specific, choose Benoit benchmark values
    "mh1" : 125.0,
    "mh2" : 120.0,
    "a2" : 0.9,
    "b4" : 1.0
}

model = SingletSM_Z2(inputParameters)

## Create Collision singleton which automatically loads the collision module
collision = WallGo.Collision()

## Optional: set the seed used by Monte Carlo integration. Default is 0
collision.setSeed(0)

## Construct a "control" object for collision integrations
collisionManager = collision.module.CollisionManager()
## TODO wrap the above too

# Use help(collisionManager) for info about what functionality is available

"""
Define couplings (Lagrangian parameters)
list as they appear in the MatrixElements file
"""
collisionManager.addCoupling(inputParameters["g3"])

"""
Define particles. 
These need masses in GeV units, ie. T dependent, but for this example we don't really have 
a temperature. So hacking this by setting T = 1. Also, for this example the vacuum mass = 0
"""
# hack
temperatureHack = 1.0
fieldHack = WallGo.Fields([0]*model.fieldCount)

"""
Register particles with the collision module. This is required for each particle that can appear in matrix elements,
including particle species that are assumed to stay in equilibrium.
The order here should be the same as in the matrix elements and how they are introduced in the model file
"""
for particle in model.particles:
    collisionManager.addParticle( constructPybindParticle(particle, temperatureHack, fieldHack) )

## Set input/output paths
scriptLocation = pathlib.Path(__file__).parent.resolve()

collisionManager.setOutputDirectory(str(scriptLocation / "CollisionOutput"))
collisionManager.setMatrixElementFile(str(scriptLocation / "MatrixElements/MatrixElements.txt"))

## Configure integration. Can skip this step if you're happy with the defaults
integrationOptions = collision.module.IntegrationOptions()
integrationOptions.bVerbose = True
integrationOptions.maxTries = 50
integrationOptions.calls = 50000
integrationOptions.relativeErrorGoal = 1e-1
integrationOptions.absoluteErrorGoal = 1e-8

collisionManager.configureIntegration(integrationOptions)

## Instruct the collision manager to print out symbolic matrix elements as it parses them. Can be useful for debugging
collisionManager.setMatrixElementVerbosity(True)

## "N". Make sure this is >= 0. The C++ code requires uint so pybind11 will throw TypeError otherwise
# basisSize = 5
basisSize = WallGo.config.getint("PolynomialGrid", "momentumGridSize")
# print(WallGo.config)

## Computes collisions for all out-of-eq particles specified above. The last argument is optional and mainly useful for debugging
collisionManager.calculateCollisionIntegrals(basisSize, bVerbose = False)