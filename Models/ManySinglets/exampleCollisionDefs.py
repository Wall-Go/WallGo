import WallGoCollision


def setupCollisionModel_QCD(
    modelParameters: dict[str, float],
) -> WallGoCollision.PhysicsModel:
    # Helper function that configures a QCD-like model for WallGoCollision

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
    gs -- QCD coupling
    msq[0] -- Mass of a fermion propagator (thermal part only, so no distinction between quark types)
    msq[1] -- Mass of a gluon propagator.

    Thermal masses depend on the QCD coupling, however the model definition always needs a numerical value for each symbol.
    This adds some complexity to the model setup, and therefore we do the symbol definitions in stages: 
    1) Define independent couplings
    2) Define helper functions for computing thermal masses from the couplings
    3) Define the mass symbols using initial values computed from the helpers.

    For purposes of the model at hand this approach is overly complicated because the mass expressions are very simple.
    However the helper functions are necessary in more general cases if using non-ultrarelativistic particle content.
    In this example the mass functions are written explicitly to demonstrate how the model setup would work in more complicated models.
    """

    # The parameter container used by WallGo collision routines is of WallGoCollision.ModelParameters type which behaves somewhat like a Python dict.
    # Here we write our parameter definitions to a local ModelParameters variable and pass it to modelDefinitions later.
    parameters = WallGoCollision.ModelParameters()

    # For defining new parameters use addOrModifyParameter(). For read-only access you can use the [] operator.
    # Here we copy the value of QCD coupling as defined in the main WallGo model (names differ for historical reasons)
    parameters.addOrModifyParameter("gs", modelParameters["g3"])

    # Define mass helper functions. We need the mass-squares in units of temperature, ie. m^2 / T^2.
    # These should take in a WallGoCollision.ModelParameters object and return a floating-point value

    # For quarks we include the thermal mass only
    def quarkThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
        gs = p["gs"]  # this is equivalent to: gs = p.getParameterValue("gs")
        return gs**2 / 6.0

    def gluonThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
        return 2.0 * p["gs"] ** 2

    parameters.addOrModifyParameter("mq2", quarkThermalMassSquared(parameters))
    parameters.addOrModifyParameter("mg2", gluonThermalMassSquared(parameters))

    # Copy the parameters to our ModelDefinition helper. This finishes the parameter part of model definition.
    modelDefinition.defineParameters(parameters)

    """Particle definitions. As described above,
    The model needs to be aware of all particles species that appear as external legs in matrix elements.
    Note that this includes also particles that are assumed to remain in equilibrium but have collisions with out-of-equilibrium particles.
    Particle definition is done by filling in a ParticleDescription struct and calling the ModelDefinition.defineParticleSpecies() method
    """
    topQuark = WallGoCollision.ParticleDescription()
    topQuark.name = "top"  # String identifier, MUST be unique
    topQuark.index = 0  # Unique integer identifier, MUST match index that appears in matrix element file
    topQuark.type = WallGoCollision.EParticleType.eFermion
    topQuark.bInEquilibrium = False
    topQuark.bUltrarelativistic = True
    topQuark.massSqFunction = quarkThermalMassSquared

    # Finish particle species definition
    modelDefinition.defineParticleSpecies(topQuark)

    ## Repeat particle definitions for light quarks and the gluon
    gluon = WallGoCollision.ParticleDescription()
    gluon.name = "gluon"
    gluon.index = 1
    gluon.type = WallGoCollision.EParticleType.eBoson
    gluon.bInEquilibrium = True
    gluon.bUltrarelativistic = True
    gluon.massSqFunction = gluonThermalMassSquared
    modelDefinition.defineParticleSpecies(gluon)

    # Light quarks remain in equilibrium but appear as external particles in collision processes, so define a generic light quark

    lightQuark = topQuark
    """Technical NOTE: Although WallGoCollision.ParticleDescription has an underlying C++ description, on Python side they are mutable objects and behave as you would expect from Python objects.
    This means that the above makes "lightQuark" a reference to the "topQuark" object, instead of invoking a copy-assignment operation on the C++ side.
    Hence we actually modify the "topQuark" object directly in the following, which is fine because the top quark definition has already been copied into the modelDefinition variable.
    """
    lightQuark.bInEquilibrium = True
    lightQuark.name = "light quark"
    lightQuark.index = 2
    modelDefinition.defineParticleSpecies(lightQuark)

    # Create the concrete model
    model = WallGoCollision.PhysicsModel(modelDefinition)
    return model
