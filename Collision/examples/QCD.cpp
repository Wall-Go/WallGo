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

#include "WallGoCollision/WallGoCollision.h"

// Configures QCD-like particle content
bool setupQCD(wallgo::CollisionManager& manager) {

	const double gs = 1.2279920495357861;

    //**** Masses squared. These need to be in units of temperature, ie. (m/T)^2 **//
	// Thermal
	const double mq2 = 0.251327; // quark, m^2 = gs^2 * T^2 / 6.0
	const double mg2 = 3.01593; // SU(3) gluon, m^2 = 2*gs^2 T^2
	// Set vacuum masses to 0
	const double msqVacuum = 0.0;

    /* Approximate all particles as ultrarelativistic, allowing heavy optimizations.
    * This means E = |p| inside collision integrations, but thermal masses are kept inside matrix element propagators.
	* Note that this may not be a good approximation for the gluons in particular due to its large thermal mass. */
	const bool bUltraRelativistic = true;

	// Take top and gluon to be out-of-equilibrium
    wallgo::ParticleSpecies topQuark("top", wallgo::EParticleType::FERMION, /*inEquilibrium*/false, msqVacuum, mq2, bUltraRelativistic);
	wallgo::ParticleSpecies gluon("gluon", wallgo::EParticleType::BOSON, false, msqVacuum, mg2, bUltraRelativistic);
	wallgo::ParticleSpecies lightQuark("quark", wallgo::EParticleType::FERMION, true, msqVacuum, mq2, bUltraRelativistic);

	// Ordering NEEDS to match the order in which particles are defined in the matrix element file(s). TODO improve this
	manager.addParticle(topQuark);
	manager.addParticle(gluon);
	manager.addParticle(lightQuark);
	

	// Define all symbol that appear in matrix elements along with an initial value. The values can be changed later with manager.setVariable(name, value)
	//manager.defineVariable("gs", gs);
	manager.defineVariable("c[0]", gs);
	manager.defineVariable("msq[0]", mq2);
	manager.defineVariable("msq[1]", mg2);
	manager.defineVariable("msq[2]", mq2);

	/* Where to load matrix elements from. If not specified, defaults to MatrixElements.txt in working dir. 
	This function returns false if the file is not found, in which case we abort here. */
	if (!manager.setMatrixElementFile("MatrixElements/MatrixElements_QCD.txt"))
	{
		std::cerr << "It seems you may be running this example program from a nonstandard location.\n"
			"The matrix elements for this example are in MatrixElements/MatrixElements_QCD.txt which is hardcoded as a relative path for simplicity.\n"
			"Please run the example program inside the 'examples' directory.\n"
			"In your own applications you can call wallgo::CollisionManager::setMatrixElementFile() to specify the file location as you prefer."
			<< std::endl;
		return false;
	}

	return true;
}


int main() 
{
	std::cout << "=== Running WallGo collision example: QCD" << std::endl;

	// We use GSL for Monte Carlo integration. It needs to be initialized before use, with optional seed (default = 0)
    wallgo::initializeRNG();

	// Can also set the seed at any later time:
	//wallgo::setSeed(42);

    wallgo::CollisionManager manager;

	// Specify output directory (relative or absolute path). Defaults to current directory
	manager.setOutputDirectory("output");

	// Setup the particle content and specify matrix element file. If the setup fails we just abort
    if (!setupQCD(manager))
	{
		return 1;
	}

	// Polynomial basis size. Using a trivially small N to make the example run fast
	const int basisSizeN = 3;
	manager.changePolynomialBasis(basisSizeN);

	/* Initialize collision integrals for all off-equilibrium particles currently registered with the manager.
	Setting verbosity to true will tell the manager to print each matrix element in a symbolic form which can be useful for debugging. */
	manager.setupCollisionIntegrals(/*verbose*/ true);

	// Configure integrator. The defaults should be reasonably OK so you can only modify what you need.
	// Here we set everything manually to show how it's done
	wallgo::IntegrationOptions options;
	options.calls = 50000;
	options.maxTries = 50;
	options.maxIntegrationMomentum = 20;
	options.absoluteErrorGoal = 1e-8;
	options.relativeErrorGoal = 1e-1;

	/* The bOptimizeUltrarelativistic flag allows the program to use a more optimized expression for the integrals
	when only particles with the ultrarelativistic flag appear as external particles.
	You should not have any reason to disable this optimization. */
	options.bOptimizeUltrarelativistic = true;
	
	manager.configureIntegration(options);

	// Evaluate all collision integrals that were prepared in the setupCollisionIntegrals() step

	std::cout << "== Evaluating collision integrals for all particles combinations ==" << std::endl;
	manager.calculateAllIntegrals(/*bVerbose*/ true);
	// TODO how to get the output from this?

	/* We can evaluate the integrals again with different model parameters, without the need to re-define particles or matrix elements.
	We can also request to compute integrals only for a specific off-equilibrium particle pair
	Demonstration: */
	std::map<std::string, double> newVars 
	{
		{"c[0]", 1},
		{"msq[1]", 0.2},
	};

	manager.setVariables(newVars);
	manager.setVariable("msq[2]", 0.3);

	std::cout << "== Evaluating (top, gluon) only ==" << std::endl;
	wallgo::CollisionTensor result = manager.evaluateCollisionTensor("top", "gluon", /*bVerbose*/ true);

	/* There is also an overloaded version of the above for passing a custom IntegrationOptions struct
	instead of using the one cached in the manager: */
	options.calls = 10000;
	result = manager.evaluateCollisionTensor("top", "gluon", options, /*bVerbose*/ false);

	// Perform clean exit
    wallgo::cleanup();
    
    return 0;
}