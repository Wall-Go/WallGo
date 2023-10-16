from lib import CollisionModule
from WallSpeed.model import Particle


## Example: create mock data in format that the C++ program can understand.
# This can be any additional input that the integral may need, eg. particle masses.

mq = 0.5013256549
mg = 1.7366430109
gs = 1.2279920495357861

data = CollisionModule.InputData()
data.values["mq"] = mq
data.values["mg"] = mg
data.values["gs"] = gs

## Call example C++ function from the module
#CollisionModule.pybindTestFunction(data)

##---- Constructing C++ ParticleSpecies class

# # print constructor usage:
#help(CollisionModule.ParticleSpecies)

# calling the constructor. Example for a top quark:
bInEquilibrium = False
bUltrarelativistic = True
## NB: needs dimensionless mass squares, ie. m^2 / T^2 
msqVacuum = 0.0
msqThermal = mq*mq

topQuark = CollisionModule.ParticleSpecies("top", CollisionModule.EParticleType.FERMION,
                                bInEquilibrium, bUltrarelativistic, msqVacuum, msqThermal)

## However the ParticleSpecies class doesn't have any functions that we could call, so it alone doesn't do much
## TODO construct collision elements here using particles as input



#CollisionModule.calculateAllCollisions()


## Convert Python 'Particle' object to pybind-bound ParticleSpecies object.
## But 'Particle' uses masses in GeV^2 units while we need m^2/T^2, so T is needed as input here
def constructPybindParticle(p : Particle, T : float):
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

    return CollisionModule.ParticleSpecies(p.name, particleType, p.inEquilibrium, p.ultrarelativistic, 
                                p.msqVacuum / T**2.0, p.msqThermal / T**2.0)