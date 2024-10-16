"""Conversion between WallGo and WallGoCollision types"""

import logging

from .particle import Particle
from .genericModel import GenericModel
from .exceptions import WallGoError

import WallGoCollision

def dictToCollisionParameters(
    inParameterDict: dict[str, float]
) -> WallGoCollision.ModelParameters:
    """Convert a python dict of named float parameters to a WallGoCollision ModelParameters object."""

    collisionParams = WallGoCollision.ModelParameters()

    for key, val in enumerate(inParameterDict):
        collisionParams.addOrModifyParameter(key, val)

    return collisionParams


def convertParticleStatistics(statisticsName: str) -> WallGoCollision.EParticleType:
    """Convert "Fermion" or "Boson" (string) to a type-safe enum.
    FIXME: Python has enums too. Use them instead of strings.
    """
    if statisticsName == "Fermion":
        return WallGoCollision.EParticleType.eFermion
    elif statisticsName == "Boson":
        return WallGoCollision.EParticleType.eBoson
    else:
        logging.warning(
            f'Invalid particle statistic: {statisticsName}. Must be "Fermion" or "Boson".'
        )
        return WallGoCollision.EParticleType.eNone


def generateCollisionParticle(
    particle: Particle, inEquilibrium: bool, ultrarelativistic: bool
) -> WallGoCollision.ParticleDescription:
    """Creates a WallGoCollision.ParticleDescription object from a WallGo.Particle.
    Note that currently this function does not support non-ultrarelativistic particles (ultrarelativistic=True raises an error). 
    """

    collisionParticle = WallGoCollision.ParticleDescription()
    collisionParticle.name = particle.name
    collisionParticle.index = particle.index
    collisionParticle.bInEquilibrium = inEquilibrium
    collisionParticle.bUltrarelativistic = ultrarelativistic
    collisionParticle.type = convertParticleStatistics(particle.statistics)

    if not ultrarelativistic:
        """Must specify mass-sq function that returns (m/T)^2. This will be used to compute energy during collision integration (E^2 = m^2 + p^2).
        Does not affect mass used in matrix element propagators,
        which has its own (user-defined) symbol and must be set in modelParameters section of collision model definition.
        
        FIXME: Currently the setup of mass functions on Python side is too different from what the collision sector needs so we can't really automate this.
        So we error out for now.
        """
        raise NotImplementedError("""Adding non-ultrarelativistic collision particles through generateCollisionParticle() is not yet supported.
                                  You can achieve this by constructing a WallGoCollision.ParticleDescription and manually defining the mass-squared function.""")
    
    return collisionParticle

## Not useful now that model params are not a first-class citizen in wallgo!
## Maybe we could have GenericModel have a getModelParameters() -> dict function that the user optionally overrides?

def generateCollisionModelDefinition(wallGoModel: GenericModel, parametersForCollisions: dict[str, float] = {}) -> WallGoCollision.ModelDefinition:
    """Automatically generates a WallGoCollision.ModelDefinition object
    with matching out-of-equilibrium particle content and model parameters as defined by the input dict.
    You will need to manually add any relevant in-equilibrium particles.
    Currently this function defines all collision particles as ultrarelativistic.

    Args:
        wallGoModel (WallGo.GenericModel):
        WallGo physics model to use as a base for the collision model.
        We take the model's outOfEquilibriumParticles list and create corresponding collision particle defs. 

        parametersForCollisions (doct[str, float]), optional:
        Dict of symbols (model parameters) that the collision model depends on
        and their current values.

        
    Returns:
        WallGoCollision.ModelDefinition:
        A partically filled collision model definition that contains all out-of-eq particles from the input model
        and has its model parameter list filled with (symbol, value) pairs as specified by the input dict. 
    """
    modelDefinition = WallGoCollision.ModelDefinition()

    for particle in wallGoModel.outOfEquilibriumParticles:
        modelDefinition.defineParticleSpecies(
            generateCollisionParticle(particle, False, True)
        ) 

    for symbol, value in parametersForCollisions.items():
        modelDefinition.defineParameter(symbol, value)

    return modelDefinition
