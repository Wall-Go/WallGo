from lib import CollisionModule
from WallSpeed.model import Particle


## Example: create mock data in format that the C++ program can understand.
# This can be any additional input that the integral may need, eg. particle masses.

mq = 0.5013256549
mg = 1.7366430109
gs = 1.2279920495357861


##---- Constructing C++ ParticleSpecies class

# # print constructor usage:
#help(CollisionModule.ParticleSpecies)

## calling the constructor. Example for a top quark:

bInEquilibrium = False
# Everything is ultrarelativistic 
bUltrarelativistic = True
## NB: needs dimensionless mass squares, ie. m^2 / T^2 
msqVacuum = 0.0
msqThermal = mq*mq



topQuark = CollisionModule.ParticleSpecies("top", CollisionModule.EParticleType.FERMION,
                                bInEquilibrium, msqVacuum, msqThermal, bUltrarelativistic)


## Make sure this is >= 0. The C++ code requires uint so pybind11 will throw TypeError otherwise
N = 20

collisionManager = CollisionModule.Collision(N)

collisionManager.addCoupling(gs)






## Convert Python 'Particle' object to pybind-bound ParticleSpecies object.
## But 'Particle' uses masses in GeV^2 units while we need m^2/T^2, so T is needed as input here.
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
    if p.statistics == "BOSON":
        particleType = CollisionModule.EParticleType.BOSON
    elif p.statistics == "FERMION":
        particleType = CollisionModule.EParticleType.FERMION

    return CollisionModule.ParticleSpecies(p.name, particleType, p.inEquilibrium, 
                                p.msqVacuum / T**2.0, p.msqThermal / T**2.0,  p.ultrarelativistic)