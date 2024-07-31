/***************************************************************************
 * WallGo/Collisions example file -- QCD.cpp.
 * Calculates collision integrals for a QCD-like theory
 * containing the SU(3) gauge field and 6 fundamental-representation quarks.
 * We allow one quark and the gluons to be out of equilibrium.
 * 
 * This setup emulates a typical electroweak phase transition wall-speed calculation
 * where the dominant friction on the bubble wall is caused by the top quark,
 * and we add collision terms of the gluons for good measure. 
 *  
****************************************************************************/

#include <filesystem>

#include "WallGo/WallGoCollision.h"

// Helper function that configures a QCD-like model
wallgo::PhysicsModel setupQCD()
{
	// Model definitions must be filled in to a ModelDefinition helper object
	wallgo::ModelDefinition modelDefinition;

	/* Specify symbolic variables that are present in matrix elements, and their initial values.
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
	However the helper functions are needed later when defining particle content anyway, see below.
	*/

	/* The parameter container used by WallGo collision routines is of wallgo::ModelParameters type, which is a wrapper around std::map.
	Here we write our parameter definitions to a ModelParameters variable and pass it to modelDefinitions later. */
	wallgo::ModelParameters parameters;

	parameters.addOrModifyParameter("gs", 1.2279920495357861);

	/* Define mass helper functions. We need the mass-squares in units of temperature, ie. m^2 / T^2.
	These should take in a wallgo::ModelParameters object and return a double value

	Here we use a C++11 lambda expression, with explicit return type, to define the mass function: */
	auto quarkThermalMassSquared = [](const wallgo::ModelParameters& params) -> double
		{
			const double gs = params.getParameterValue("gs");
			return gs * gs / 6.0;
		};

	auto gluonThermalMassSquared = [](const wallgo::ModelParameters& params) -> double
		{
			const double gs = params.getParameterValue("gs");
			return 2.0 * gs * gs;
		};

	parameters.addOrModifyParameter("msq[0]", quarkThermalMassSquared(parameters));
	parameters.addOrModifyParameter("msq[1]", gluonThermalMassSquared(parameters));

	modelDefinition.defineParameters(parameters);

	/*** Particle definitions.
	* The model needs to be aware of all particles species that appear as external legs in matrix elements.
	* Note that this includes also particles that are assumed to remain in equilibrium but have collisions with out-of-equilibrium particles.
	* Particle definition is done by filling in a ParticleDescription struct and calling ModelDefinition::defineParticleSpecies. */
	wallgo::ParticleDescription topQuark;
	topQuark.name = "top"; // Ttring identifier, MUST be unique
	topQuark.index = 0; // Unique integer identifier, MUST match index that appears in matrix element file
	topQuark.type = wallgo::EParticleType::eFermion; // Statistics: boson or fermion
	topQuark.bInEquilibrium = false; // Whether the particle species is assumed to remain in equilibrium or not

	/* For each particle you can specify how its energy should be calculated during collision integration.
	* In general the dispersion relation is E^2 = p^2 + m^2, where p is the 3-momentum, and the mass will be discussed shortly.
	* Flagging a particle as ultrarelativistic means the dispersion relation is simply E(p) = |p|, which is a valid approximation at leading log order.
	* WallGo is able to heavily optimize collision integrations if all particles are treated as ultrarelativistic,
	* so it's generally adviced to use this feature unless the ultrarelativistic approximation does not work well for your particle. */
	topQuark.bUltrarelativistic = true;

	/* We must also specify a function that computes the mass-squared of the particle from given ModelParameters input.
	* This mass will be used in the energy dispersion relation as described above.
	* Note in particular that the mass defined here is DIFFERENT from the propagator masses we defined above as "model parameters".
	* In many model setups they may be equal, but WallGo does not enforce this.
	* This setting can be skipped if the particle is ultrarelativistic; here we include it for completeness. */

	/* massSqFunction must be a callable function with signature f : ModelParameters -> double
	* and should return mass squared in units of the temperature, ie. m^2 / T^2.
	* We already defined a helper function for computing the thermal quark mass from model parameters, so we simply pass that function here. */
	topQuark.massSqFunction = quarkThermalMassSquared;

	// Finish particle definition and make the model aware of this particle species:
	modelDefinition.defineParticleSpecies(topQuark);

	/* Repeat particle definitions for light quarks and the gluon. */

	wallgo::ParticleDescription gluon;
	gluon.name = "gluon";
	gluon.index = 1;
	gluon.type = wallgo::EParticleType::eBoson;
	gluon.bInEquilibrium = false;
	gluon.bUltrarelativistic = true;
	gluon.massSqFunction = gluonThermalMassSquared;

	modelDefinition.defineParticleSpecies(gluon);

	// Light quarks remain in equilibrium but appear as external particles in collision processes, so define a generic light quark:
	wallgo::ParticleDescription lightQuark = topQuark;
	lightQuark.name = "light quark";
	lightQuark.index = 2;
	lightQuark.bInEquilibrium = true;

	modelDefinition.defineParticleSpecies(lightQuark);

	// Create the concrete model
	wallgo::PhysicsModel model(modelDefinition);

	/* Read and verify matrix elements from a file. This is done with PhysicsModel::readMatrixElements.
	Note that the model will only read matrix elements that are relevant for its out-of-equilibrium particle content. */

	/* Where to load matrix elements from. In this example the path is hardcoded relative to the working directory for simplicity. */
	std::filesystem::path matrixElementFile = "MatrixElements/MatrixElements_QCD.txt";

	if (!std::filesystem::exists(matrixElementFile))
	{
		std::cerr << "It seems you may be running this example program from a nonstandard location.\n"
			"The matrix elements for this example are in MatrixElements/MatrixElements_QCD.txt which is hardcoded as a relative path for simplicity.\n"
			"Please run the example program inside the 'examples' directory.\n"
			<< std::endl;
	}

	// Should we print each parsed matrix element to stdout? Can be useful for logging and debugging purposes
	bool bPrintMatrixElements = true;

	bool bMatrixElementsOK = model.readMatrixElements(matrixElementFile, bPrintMatrixElements);
	
	// If something in matrix element parsing went wrong we abort here
	if (!bMatrixElementsOK)
	{
		std::cerr << "\nCRITICAL: Matrix element parsing failed, aborting!" << std::endl;
		std::exit(EXIT_FAILURE);
	}

	return model;
}


int main() 
{
	std::cout << "### Running WallGo collision example : QCD ###" << std::endl;

	/* We use Monte Carlo for collision integrations, which needs random numbers.
	You must initialize the RNG before use, with optional seed (default = 0) */ 
    wallgo::initializeRNG();

	// Can also set the seed at any later time. Example:
	//wallgo::setSeed(42);

	/* First step is model definition, which specifies particle content and model parameters relevant for collisions.
	See the helper function above for details, which also prepares matrix elements.
	Note that it is NOT possible to change model particle or parameter content after creation, only parameter value changes are allowed; see below for an example. */
	wallgo::PhysicsModel model = setupQCD();

	// Polynomial basis size. Using a trivially small N to make the example run fast
	const int basisSizeN = 3;

	/* The CollisionTensor class acts as a main interface into collision integral computations.
	It will be linked to the PhysicsModel that creates it, so that changes to model parameters and particles
	will directly propagate to CollisionTensor objects created from it.
	This also means that you MUST keep the model alive for as long as you use CollisionTensors linked to it. */
	wallgo::CollisionTensor collisionTensor = model.createCollisionTensor(basisSizeN);

	// Can change the basis size easily without having to reconstruct the tensor. Example:
	//collisionTensor.changePolynomialBasisSize(7);

	/* Configure integrator. The defaults should be reasonably OK so you can only modify what you need.
	Here we set everything manually to show how it's done. */
	wallgo::IntegrationOptions integrationOptions;
	integrationOptions.calls = 50000;
	integrationOptions.maxTries = 50;
	integrationOptions.maxIntegrationMomentum = 20; // collision integration momentum goes from 0 to maxIntegrationMomentum. This is in units of temperature
	integrationOptions.absoluteErrorGoal = 1e-8;
	integrationOptions.relativeErrorGoal = 1e-1;
	
	// Override the built-in defaults with our new settings
	collisionTensor.setDefaultIntegrationOptions(integrationOptions);

	/* We can also configure various verbosity settings. These include progress reporting and time estimates
	as well as a full result dump of each individual integral to stdout. By default these are all disabled.
	Here we enable some for demonstration purposes */
	wallgo::CollisionTensorVerbosity verbosity;
	verbosity.bPrintElapsedTime = true; // report total time when finished with all integrals

	/* Progress report when this percentage of total integrals (approximately) have been computed.
	Our example has so few integrals that this is quite silly,
	and in multithreaded context it can activate much less frequently than every 25% because the reporting is always done from the "main" thread.
	Note that this percentage is per-particle-pair, ie. each (particle1, particle2) pair reports when this percentage of their own integrals is done. */
	verbosity.progressReportPercentage = 0.25;

	// Very slow and verbose, intended only for debugging purposes
	verbosity.bPrintEveryElement = true; 

	// Override the built-in defaults with our new settings
	collisionTensor.setDefaultIntegrationVerbosity(verbosity);

	// Evaluate all collision integrals that were prepared in the setupCollisionIntegrals() step
	std::cout << "== Evaluating collision integrals for all particles combinations ==" << std::endl;
	wallgo::CollisionTensorResult results = collisionTensor.computeIntegralsAll();

	/* Write results to disk using HDF5 format. Each particle pair gets its own HDF5 file.
	The bool argument specifies whether statistical errors should be written as well.
	Statistical errors will go to a separate dataset in the HDF5 file. */
	results.writeToIndividualHDF5(/*output directory*/ "output", /*bWriteErrors*/ true);
	
	/* We can evaluate the integrals again with different model parameters, without the need to re-define particles or matrix elements.
	As explained above, CollisionTensors are linked to their respective PhysicsModel objects, so we just need to modify the model object to achieve this.
	Note we use the PhysicsModel::updateParameter() method to modify existing parameters and not PhysicsModel::defineParameter(). */
	model.updateParameter("gs", 0.5); // some random value

	// Can also pack the new parameters in a wallgo::ModelParameters object and pass it to the model:
	wallgo::ModelParameters changedParams;
	changedParams.addOrModifyParameter("gs", 0.5);
	changedParams.addOrModifyParameter("msq[1]", 0.3);
	model.updateParameters(changedParams);

	/* We can also request to compute integrals only for a specific off-equilibrium particle pair.
	Demonstration: */
	std::cout << "== Evaluating (top, top) only with modified parameters ==" << std::endl;
	wallgo::CollisionResultsGrid resultsTopGluon = collisionTensor.computeIntegralsForPair("top", "top");

	/* CollisionTensor also defines overloaded versions of the main "compute" functions for specifying custom
	IntegrationOptions and CollisionTensorVerbosity objects on a per-call basis, instead defaulting to the ones cached inside the CollisionTensor instance.
	Can be used for more fine-grained evaluation.
	Demonstration: */
	integrationOptions.calls = 10000;
	verbosity.bPrintEveryElement = false;
	verbosity.progressReportPercentage = 0; // no progress reporting
	verbosity.bPrintElapsedTime = true;
	std::cout << "== Evaluating (top, top) only without progress tracking ==" << std::endl;
	resultsTopGluon = collisionTensor.computeIntegralsForPair("top", "top", integrationOptions, verbosity);

	// Perform clean exit
    wallgo::cleanup();
    
    return 0;
}
