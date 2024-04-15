"""Simple example file for using the WallGo collision module from Python.
The module is written in C++ but implements Python bindings. There is also a pure Python wrapper class,
WallGo.Collision, which handles dynamic loading of the module and provides complementary functionality.
Note that WallGo.Collision is a singleton class, ie. only one instance of it can exist.
Loading of the module happens when the instance is first created."""

import pathlib

import WallGo
from WallGo import Particle



## TODO move this to the collision wrapper:

## Convert Python 'Particle' object to pybind-bound ParticleSpecies object.
## But 'Particle' uses masses in GeV^2 units while we need m^2/T^2, so T is needed as input here.
## Should do the same for field values since the vacuum mass can depend on that.
## Return value is a ParticleSpecies object
def constructPybindParticle(p: Particle, T: float):
    r"""
        Converts 'Particle' object to ParticleSpecies object that the Collision module can understand.
        CollisionModule operates with dimensionless (m/T)^2 etc, so the temperature is taken as an input here. 

        Parameters
        ----------
        p : Particle
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
    if p.statistics == "Boson":
        particleType = WallGo.Collision().module.EParticleType.BOSON
    elif p.statistics == "Fermion":
        particleType =  WallGo.Collision().module.EParticleType.FERMION

    return WallGo.Collision().module.ParticleSpecies(p.name, particleType, p.inEquilibrium, 
                                p.msqVacuum / T**2.0, p.msqThermal(T) / T**2.0,  p.ultrarelativistic)



WallGo.initialize()

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
"""
gs = 1.2279920495357861

collisionManager.addCoupling(gs)

"""
Define particles. 
These need masses in GeV units, ie. T dependent, but for this example we don't really have 
a temperature. So hacking this by setting T = 1. Also, for this example the vacuum mass = 0
"""

topQuark = Particle(
    name="top",
    msqVacuum=0.0,
    msqDerivative = 0.0,
    msqThermal=lambda T: 0.251327 * T**2,
    statistics="Fermion",
    inEquilibrium=False,
    ultrarelativistic=True,
    totalDOFs = 12
)

gluon = Particle(
    name="gluon",
    msqVacuum=0.0,
    msqDerivative = 0.0,
    msqThermal=lambda T: 3.01593 * T**2,
    statistics="Boson",
    inEquilibrium=True,
    ultrarelativistic=True,
    totalDOFs = 16
)

lightQuark = Particle(
    name="quark",
    msqVacuum=0.0,
    msqDerivative = 0.0,
    msqThermal=lambda T: 0.251327 * T**2,
    statistics="Fermion",
    inEquilibrium=True,
    ultrarelativistic=True,
    totalDOFs = 60
)

# hack
temperatureHack = 1.0

"""Register particles with the collision module. This is required for each particle that can appear in matrix elements,
including particle species that are assumed to stay in equilibrium."""

collisionManager.addParticle( constructPybindParticle(topQuark, temperatureHack) )
collisionManager.addParticle( constructPybindParticle(gluon, temperatureHack) )
collisionManager.addParticle( constructPybindParticle(lightQuark, temperatureHack) )

## Set input/output paths
scriptLocation = pathlib.Path(__file__).parent.resolve()

collisionManager.setOutputDirectory(str(scriptLocation / "CollisionOutput"))
collisionManager.setMatrixElementFile(str(scriptLocation / "MatrixElements/MatrixElements_QCD.txt"))

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
basisSize = 5

## Computes collisions for all out-of-eq particles specified above. The last argument is optional and mainly useful for debugging
collisionManager.calculateCollisionIntegrals(basisSize, bVerbose = False)