import WallGoCollision


def setupCollisionModel_Yukawa(
    modelParameters: dict[str, float],
) -> WallGoCollision.PhysicsModel:
    # Helper function that configures the Yukawa model

    """Model definitions must be filled in to a ModelDefinition helper struct.
    This has two main parts:
    1) Model parameter definitions which must contain all model-specific symbols that appear in matrix elements. This must include any particle masses that appear in propagators.
    2) List of all particle species that appear as external legs in collision processes. If ultrarelativistic approximations are NOT used for some particle species,
    the particle definition must contain a function that computes the particle mass from model parameters. This must be defined even if you include mass variables in the model parameters in stage 1).
    Details of both stages are described in detail below.
    """
    modelDefinition = WallGoCollision.ModelDefinition()

    """Specify symbolic variables that are present in matrix elements, and their initial values.
    This typically includes at least coupling constants of the theory, but often also masses of fields that appear in internal propagators.
    Depending on your model setup, the propagator masses may or may not match with "particle" masses used elsewhere in WallGo.

    In this example the symbols needed by matrix elements are:
    y -- The Yukawa coupling
    gamma -- The scalar cubic interaction 
    lam -- The scalar quartic interaction
    mf2 -- Mass of a fermion propagator (thermal part only)
    ms2 -- Mass of a scalar propagator.
    """

    # The parameter container used by WallGo collision routines is of WallGoCollision.ModelParameters type which behaves somewhat like a Python dict.
    # Here we write our parameter definitions to a local ModelParameters variable and pass it to modelDefinitions later.
    parameters = WallGoCollision.ModelParameters()

    # For defining new parameters use addOrModifyParameter(). For read-only access you can use the [] operator.
    # Here we copy the value of the couplings as defined in the main WallGo model
    parameters.add("y", modelParameters["y"])
    parameters.add("gamma", modelParameters["gamma"])
    parameters.add("lam", modelParameters["lam"])

    # Define mass helper functions. We need the mass-squares in units of temperature, ie. m^2 / T^2.
    # These should take in a WallGoCollision.ModelParameters object and return a floating-point value

    # Thermal mass for the fermions
    def fermionThermalMass(p: WallGoCollision.ModelParameters) -> float:
        return 1 / 16 * p["y"] ** 2

    def scalarThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
        return p["lam"] / 24.0 + p["y"] ** 2.0 / 6.0

    parameters.add("mf2", fermionThermalMass(parameters))
    parameters.add("ms2", scalarThermalMassSquared(parameters))

    # Copy the parameters to our ModelDefinition helper. This finishes the parameter part of model definition.
    modelDefinition.defineParameters(parameters)

    # Particle definitions
    # Add in-equilibrium particles that appear in collision processes
    phiParticle = WallGoCollision.ParticleDescription()
    phiParticle.name = "phi"
    phiParticle.index = 0
    phiParticle.bInEquilibrium = True
    phiParticle.bUltrarelativistic = True
    phiParticle.type = WallGoCollision.EParticleType.eBoson

    modelDefinition.defineParticleSpecies(phiParticle)


    # Create the concrete model
    model = WallGoCollision.PhysicsModel(modelDefinition)
    return model
