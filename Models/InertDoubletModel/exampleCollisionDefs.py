import WallGoCollision


def setupCollisionModel_IDM(
    modelParameters: dict[str, float],
) -> WallGoCollision.PhysicsModel:
    # Helper function that configures a model with QCD and ElectroWeak interactions for WallGoCollision

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
    gw -- electroweak couplings
    lam3H -- portal couplings between the Higgs and the inert doublet
    yt1 -- Yukawa coupligs
    mq2 -- Mass of a fermion propagator (thermal part only, and only QCD-contribution to thermal mass, so no distinction between quark types)
    mg2 -- Mass of a gluon propagator.
    mw2 -- Mass of a W propagator.
    mh2 -- Mass of a Higgs propagator.
    mA2 -- Mass of a A propogator
    mH2 -- Mass of a H propagator
    ml2 -- Mass of a lepton propagator

    Thermal masses depend on the couplings, however the model definition always needs a numerical value for each symbol.
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
    # Here we copy the value of QCD, EW and scalar couplings as defined in the main WallGo model (names differ for historical reasons)
    parameters.add("g3", modelParameters["g3"])
    parameters.add("gw", modelParameters["g2"])
    parameters.add("g1", modelParameters["g1"])
    parameters.add("yt1", modelParameters["yt"])
    parameters.add("lam1H", modelParameters["lambda"])
    parameters.add("lam3H", modelParameters["lambda3"])
    parameters.add("lam4H", modelParameters["lambda4"])
    parameters.add("v", modelParameters["vevCollisions"])

    # Define mass helper functions. We need the mass-squares in units of temperature, ie. m^2 / T^2.
    # These should take in a WallGoCollision.ModelParameters object and return a floating-point value

    # For quarks we include the thermal mass only
    def quarkThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
        gs = p["g3"]  # this is equivalent to: gs = p.getParameterValue("gs")
        return gs**2 / 6.0
    
    def leptonThermalMassSquared(p :WallGoCollision.ModelParameters) -> float:
        return 3* p["gw"]**2 / 32.0

    def gluonThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
        return 2.0 * p["g3"] ** 2
    
    def wBosonThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
        return 11.0 * p["gw"] ** 2 / 6.0

    def HiggsBosonThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
        return (6* p["lam1H"]+ 2 * p["lam3H"] +p["lam4H"]+ 3*(3 * p["gw"] ** 2 + p["g1"]**2) / 4 + 3 * p["yt1"]**2) / 12.0
    
    def newScalarThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
        return p["lam3H"] / 24.0


    parameters.add("mq2", quarkThermalMassSquared(parameters))
    parameters.add("ml2", leptonThermalMassSquared(parameters))
    parameters.add("mg2", gluonThermalMassSquared(parameters))
    parameters.add("mw2", wBosonThermalMassSquared(parameters))
    parameters.add("mh2", HiggsBosonThermalMassSquared(parameters))
    parameters.add("mH2", newScalarThermalMassSquared(parameters))
    parameters.add("mA2", newScalarThermalMassSquared(parameters))


    # Copy the parameters to our ModelDefinition helper. This finishes the parameter part of model definition.
    modelDefinition.defineParameters(parameters)

    
    # Particle definitions
    topQuarkL = WallGoCollision.ParticleDescription()
    topQuarkL.name = "TopL"  # String identifier, MUST be unique
    topQuarkL.index = 0  # Unique integer identifier, MUST match index that appears in matrix element file
    topQuarkL.type = (
        WallGoCollision.EParticleType.eFermion
    )  # Statistics (enum): boson or fermion
    topQuarkL.bInEquilibrium = (
        False  # Whether the particle species is assumed to remain in equilibrium or not
    )
    topQuarkL.bUltrarelativistic = True
    topQuarkL.massSqFunction = quarkThermalMassSquared
    modelDefinition.defineParticleSpecies(topQuarkL)

    topQuarkR = WallGoCollision.ParticleDescription()
    topQuarkR.name = "TopR"
    topQuarkR.index = 1
    topQuarkR.type = WallGoCollision.EParticleType.eFermion
    topQuarkR.bInEquilibrium = False
    topQuarkR.bUltrarelativistic = True
    topQuarkR.massSqFunction = quarkThermalMassSquared
    modelDefinition.defineParticleSpecies(topQuarkR)

    lightQuark = WallGoCollision.ParticleDescription()
    lightQuark.name = "LightQuark"
    lightQuark.index = 2
    lightQuark.type = WallGoCollision.EParticleType.eFermion
    lightQuark.bInEquilibrium = True
    lightQuark.bUltrarelativistic = True
    lightQuark.massSqFunction = quarkThermalMassSquared
    modelDefinition.defineParticleSpecies(lightQuark)

    gluon = WallGoCollision.ParticleDescription()
    gluon.name = "Gluon"
    gluon.index = 3
    gluon.type = WallGoCollision.EParticleType.eBoson
    gluon.bInEquilibrium = True
    gluon.bUltrarelativistic = True
    gluon.massSqFunction = gluonThermalMassSquared
    modelDefinition.defineParticleSpecies(gluon)

    wBoson = WallGoCollision.ParticleDescription()
    wBoson.name = "W"
    wBoson.index = 4
    wBoson.type = WallGoCollision.EParticleType.eBoson
    wBoson.bInEquilibrium = False
    wBoson.bUltrarelativistic = True
    wBoson.massSqFunction = wBosonThermalMassSquared
    modelDefinition.defineParticleSpecies(wBoson)

    higgsDoublet = WallGoCollision.ParticleDescription() #Contains also Goldstones
    higgsDoublet.name = "Higgs"
    higgsDoublet.index = 5
    higgsDoublet.type = WallGoCollision.EParticleType.eBoson
    higgsDoublet.bInEquilibrium = True
    higgsDoublet.bUltrarelativistic = True
    higgsDoublet.massSqFunction = HiggsBosonThermalMassSquared
    modelDefinition.defineParticleSpecies(higgsDoublet)

    newScalarHeavy = WallGoCollision.ParticleDescription()
    newScalarHeavy.name = "A"
    newScalarHeavy.index = 6
    newScalarHeavy.type = WallGoCollision.EParticleType.eBoson
    newScalarHeavy.bInEquilibrium = False
    newScalarHeavy.bUltrarelativistic = True
    newScalarHeavy.massSqFunction = newScalarThermalMassSquared
    modelDefinition.defineParticleSpecies(newScalarHeavy)

    # Create the concrete model
    model = WallGoCollision.PhysicsModel(modelDefinition)
    return model
