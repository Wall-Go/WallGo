"""Conversion between WallGo and WallGoCollision types"""

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
        print(
            f'Invalid particle statistic: {statisticsName}. Must be "Fermion" or "Boson".'
        )
        return WallGoCollision.EParticleType.eNone


def particleToCollisionParticleDescription(
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
        raise NotImplementedError("""Adding non-ultrarelativistic collision particles through particleToCollisionParticleDescription() is not yet supported.
                                  You can achieve this by constructing a WallGoCollision.ParticleDescription and manually defining the mass-squared function.""")
    
    return collisionParticle


def generateCollisionModelDefinition(wallGoModel: GenericModel, includeAllModelParameters: bool = True, parameterSymbolsToInclude: list[str] = []) -> WallGoCollision.ModelDefinition:
    """Automatically generates a WallGoCollision.ModelDefinition object
    with matching out-of-equilibrium particle content and model parameters as the input WallGo.GenericModel object.
    You will need to manually add any relevant in-equilibrium particles and parameters that the collision terms depend on.
    Currently this function defines all collision particles as ultrarelativistic.

    Args:
        wallGoModel (WallGo.GenericModel):
        WallGo physics model to use as a base for the collision model.
        We take the model's outOfEquilibriumParticles list and create corresponding collision particle defs. 

        includeAllModelParameters (bool):
        If True, the produced collision model definition will depend on all symbols
        contained in the input model's modelParameters dict.

        parameterSymbolsToInclude (list[str]), optional:
        List of symbols (model parameters) that the collision model depends on.
        The input model must have defined a value for each symbol in its modelParameters dict.
        This argument is ignored if includeAllModelParameters is set to True.

        
    Returns:
        WallGoCollision.ModelDefinition:
        A partically filled collision model definition that contains all out-of-eq particles from the input model
        and has its model parameter list filled with symbols as specified by the other argumetns. 
    """
    modelDefinition = WallGoCollision.ModelDefinition()

    for particle in wallGoModel.outOfEquilibriumParticles:
        modelDefinition.defineParticleSpecies(
            particleToCollisionParticleDescription(particle, False, True)
            ) 
    
    if includeAllModelParameters:
        modelDefinition.defineParameters(dictToCollisionParameters(wallGoModel.modelParameters))

    else:
        # Define just the symbols in the input list
        for symbol in parameterSymbolsToInclude:
            if symbol not in wallGoModel.modelParameters:
                raise WallGoError(f"Symbol {symbol} not found in the input WallGo model")
            
            else:
                modelDefinition.defineParameter(symbol, wallGoModel.modelParameters.get(symbol))

    return modelDefinition
