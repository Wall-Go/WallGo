
from WallSpeed.CollisionModuleLoader import CollisionModule
from WallSpeed import Particle



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
        particleType = CollisionModule.EParticleType.BOSON
    elif p.statistics == "Fermion":
        particleType = CollisionModule.EParticleType.FERMION

    return CollisionModule.ParticleSpecies(p.name, particleType, p.inEquilibrium, 
                                p.msqVacuum / T**2.0, p.msqThermal(T) / T**2.0,  p.ultrarelativistic)


## Make sure this is >= 0. The C++ code requires uint so pybind11 will throw TypeError otherwise
polynomialBasisSize = 20    # "N"

## Construct a "control" object for collision integrations
collisionManager = CollisionModule.Collision(polynomialBasisSize)



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

## @todo probably get rid of the collisionPrefactors variable? CollisionModule doesn't need anything like it

topQuark = Particle(
    name="top",
    msqVacuum=0.0,
    msqThermal=lambda T: 0.251327 * T**2,
    statistics="Fermion",
    inEquilibrium=False,
    ultrarelativistic=True,
    collisionPrefactors=[0, 0, 0],
)

lightQuark = Particle(
    name="quark",
    msqVacuum=0.0,
    msqThermal=lambda T: 0.251327 * T**2,
    statistics="Boson",
    inEquilibrium=True,
    ultrarelativistic=True,
    collisionPrefactors=[0, 0, 0],
)

gluon = Particle(
    name="gluon",
    msqVacuum=0.0,
    msqThermal=lambda T: 3.01593 * T**2,
    statistics="Boson",
    inEquilibrium=True,
    ultrarelativistic=True,
    collisionPrefactors=[0, 0, 0],
)

# hack
T = 1.0

collisionManager.addParticle( constructPybindParticle(topQuark, T) )
collisionManager.addParticle( constructPybindParticle(lightQuark, T) )
collisionManager.addParticle( constructPybindParticle(gluon, T) )

## TODO specify file name where the matrix elements will be read from. Right now 
## the class has these hardcoded in

collisionManager.calculateCollisionIntegrals()