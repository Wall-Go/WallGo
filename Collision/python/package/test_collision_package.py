"""This is a Python counterpart of QCD.cpp example file to demonstrate the use of Python bindings of the WallGo collision module.
This is intended as a verbose example for displaying the features available to the Python side.
"""

import sys
import os
from pathlib import Path

try:
    # NB: package in working dir gets priority over whatever global installation
    import WallGoCollision
    print(f"Loaded WallGoCollision package from location: {WallGoCollision.__path__}")
    
except Exception as e:
    print(f"Error: {e}")
    input("Press any key to exit...")
    sys.exit(1)

#WallGoCollision.setSeed() ## crash??

# Helper function that configures a QCD-like model for WallGoCollision
def setupQCD() -> WallGoCollision.PhysicsModel:

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

    For purposes of this example this approach is overly explicit because the mass expressions are very simple.
    However the helper functions are needed later when defining particle content anyway (assuming non-ultrarelativistic particles), see below.
    """

    # The parameter container used by WallGo collision routines is of WallGoCollision.ModelParameters type which behaves somewhat like a Python dict.
    # Here we write our parameter definitions to a local ModelParameters variable and pass it to modelDefinitions later.
    parameters = WallGoCollision.ModelParameters()

    # For defining new parameters use addOrModifyParameter(). For read-only access you can use the [] operator
    parameters.addOrModifyParameter("gs", 1.2279920495357861)

    # Define mass helper functions. We need the mass-squares in units of temperature, ie. m^2 / T^2.
    # These should take in a WallGoCollision.ModelParameters object and return a floating-point value

    # For quarks we include the thermal mass only
    def quarkThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
        gs = p["gs"] # this is equivalent to: gs = p.getParameterValue("gs")
        return gs**2 / 6.0 

    def gluonThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
        return 2.0 * p["gs"]**2

    parameters.addOrModifyParameter("msq[0]", quarkThermalMassSquared(parameters))
    parameters.addOrModifyParameter("msq[1]", gluonThermalMassSquared(parameters))  

    # Copy the parameters to our ModelDefinition helper. This finishes the parameter part of model definition.
    modelDefinition.defineParameters(parameters)

    """Particle definitions. As described above,
    The model needs to be aware of all particles species that appear as external legs in matrix elements.
    Note that this includes also particles that are assumed to remain in equilibrium but have collisions with out-of-equilibrium particles.
    Particle definition is done by filling in a ParticleDescription struct and calling the ModelDefinition.defineParticleSpecies() method
    """
    topQuark = WallGoCollision.ParticleDescription()
    topQuark.name = "top" # string identifier, MUST be unique
    topQuark.index = 0 # Unique integer identifier, MUST match index that appears in matrix element file
    topQuark.type = WallGoCollision.EParticleType.eFermion # Statistics (enum): boson or fermion
    topQuark.bInEquilibrium = False # Whether the particle species is assumed to remain in equilibrium or not
    """For each particle you can specify how its energy should be calculated during collision integration.
    In general the dispersion relation is E^2 = p^2 + m^2, where p is the 3-momentum, and the mass will be discussed shortly.
    Flagging a particle as ultrarelativistic means the dispersion relation is simply E(p) = |p|, which is a valid approximation at leading log order.
    WallGo is able to heavily optimize collision integrations if all particles are treated as ultrarelativistic,
    so it's generally adviced to use this feature unless your accuracy goal excludes the ultrarelativistic approximation.
    """
    topQuark.bUltrarelativistic = True

    """We must also specify a function that computes the mass-squared of the particle from given ModelParameters input.
    This mass will be used in the energy dispersion relation as described above and can be skipped for ultrarelativistic particles. Here we include it for completeness.
    Note in particular that the mass defined here is DIFFERENT from the propagator masses we defined above as "model parameters".
    In many model setups they may be equal, but WallGoCollision does not enforce this.
    
    massSqFunction must be a callable function with signature f: WallGoCollision.ModelParameters -> float
    and should return mass squared in units of the temperature, ie. m^2 / T^2.
    We already defined a helper function for computing the thermal quark mass from model parameters, so here we simply pass that function.
    """
    topQuark.massSqFunction = quarkThermalMassSquared

    # Finish particle species definition
    modelDefinition.defineParticleSpecies(topQuark)
    
    ## Repeat particle definitions for light quarks and the gluon
    gluon = WallGoCollision.ParticleDescription()
    gluon.name = "gluon"
    gluon.index = 1
    gluon.type = WallGoCollision.EParticleType.eBoson
    gluon.bInEquilibrium = False
    gluon.bUltrarelativistic = True
    gluon.massSqFunction = gluonThermalMassSquared
    modelDefinition.defineParticleSpecies(gluon)

    # Light quarks remain in equilibrium but appear as external particles in collision processes, so define a generic light quark

    lightQuark = topQuark
    """Technical NOTE: Although WallGoCollision.ParticleDescription has an underlying C++ description, on Python side they are mutable objects and behave as you would expect from Python objects.
    This means that the above makes "lightQuark" a reference to the "topQuark" object, instead of invoking a copy-assignment operation on the C++ side.
    Hence we actually modify the "topQuark" object directly in the following, which is fine because the top quark definition has already been fixed and copied into the modelDefinition variable.
    """
    lightQuark.bInEquilibrium = True
    lightQuark.name = "light quark"
    lightQuark.index = 2
    modelDefinition.defineParticleSpecies(lightQuark)
    
    # Create the concrete model
    model = WallGoCollision.PhysicsModel(modelDefinition)

    ## Read and verify matrix elements from a file. This is done with PhysicsModel.readMatrixElements().
	## Note that the model will only read matrix elements that are relevant for its out-of-equilibrium particle content.

	# Where to load matrix elements from. In this example the path is hardcoded relative to the script directory for simplicity.
    scriptDir = os.path.dirname(__file__)
    matrixElementsFile = Path(scriptDir) / "../../examples/MatrixElements/MatrixElements_QCD.txt"

    if not matrixElementsFile.is_file():
        print("""It seems you may be running this example script from a nonstandard location.""")
        ## TODO better pathing

    # Should we print each parsed matrix element to stdout? Can be useful for logging and debugging purposes
    bPrintMatrixElements = True

    # The path needs to be passed as a string
    bMatrixElementsOK = model.readMatrixElements(str(matrixElementsFile), bPrintMatrixElements);
	
    # If something in matrix element parsing went wrong we abort here
    if not bMatrixElementsOK:
        print("CRITICAL: Matrix element parsing failed, aborting!")
        sys.exit(1)

    return model

def collisionExampleQCD() -> None:
    
    """First step is model definition, which specifies particle content and model parameters relevant for collisions.
	See the helper function above for details, which also prepares matrix elements.
	Note that it is NOT possible to change model particle or parameter content after creation, only parameter value changes are allowed.
    See below for an example.
    """
    model = setupQCD()
    
    # Polynomial basis size. Using a trivially small N to make the example run fast
    basisSizeN = 3

    """The CollisionTensor class acts as a main interface into collision integral computations.
    It will be linked to the PhysicsModel that creates it, so that changes to model parameters
    will directly propagate to CollisionTensor objects created from it.
    This also means that you MUST keep the model alive for as long as you use CollisionTensors linked to it.
    """
    collisionTensor = model.createCollisionTensor(basisSizeN)
    
    # Can change the basis size easily without having to reconstruct the tensor. Example:
    #collisionTensor.changePolynomialBasisSize(7)

    """Configure the integrator. Default settings should be reasonably OK so you can modify only what you need,
    or skip this step entirely. Here we set everything manually to show how it's done.
    """
    integrationOptions = WallGoCollision.IntegrationOptions()
    integrationOptions.calls = 50000
    integrationOptions.maxTries = 50
    integrationOptions.maxIntegrationMomentum = 20 # collision integration momentum goes from 0 to maxIntegrationMomentum. This is in units of temperature
    integrationOptions.absoluteErrorGoal = 1e-8
    integrationOptions.relativeErrorGoal = 1e-1
    
    collisionTensor.setIntegrationOptions(integrationOptions)
    
    """We can also configure various verbosity settings that are useful when you want to see what is going on in long-running integrations.
    These include progress reporting and time estimates, as well as a full result dump of each individual integral to stdout.
    By default these are all disabled. Here we enable some for demonstration purposes.
    """
    verbosity = WallGoCollision.CollisionTensorVerbosity()
    verbosity.bPrintElapsedTime = True # report total time when finished with all integrals

    """Progress report when this percentage of total integrals (approximately) have been computed.
    Our example has so few integrals that this is quite silly,
    and in multithreaded context it can activate much less frequently than every 25% because the reporting is always done from the "main" thread.
    Note that this percentage is per-particle-pair, ie. each (particle1, particle2) pair reports when this percentage of their own integrals is done.
    """
    verbosity.progressReportPercentage = 0.25

	# Very slow and verbose, intended only for debugging purposes
    verbosity.bPrintEveryElement = True
    
    collisionTensor.setIntegrationVerbosity(verbosity)

    print(f"== Evaluating collision integrals for all particle combinations ==")
    # Full evaluation of the collision tensor. This is where the main work is done. Return type is CollisionTensorResult which is a convenience wrapper around a numerical array
    results: WallGoCollision.CollisionTensorResult = collisionTensor.computeIntegralsAll()

    


if __name__ == "__main__":
    collisionExampleQCD()