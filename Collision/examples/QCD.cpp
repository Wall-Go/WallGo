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
	const double mq2 = 0.251327; // quark
	const double mg2 = 3.01593; // SU(3) gluon
	// Set vacuum masses to 0 
	const double msqVacuum = 0.0;

    /* Approximate all particles as ultrarelativistic, allowing heavy optimizations.
    * This means E = |p| inside collision integrations, but the masses are kept inside matrix element propagators.
    */
	const bool bUltraRelativistic = true;

	// Take top and gluon to be out-of-equilibrium
    wallgo::ParticleSpecies topQuark("top", wallgo::EParticleType::FERMION, /*inEquilibrium*/false, msqVacuum, mq2, bUltraRelativistic);
    wallgo::ParticleSpecies gluon("gluon", wallgo::EParticleType::BOSON, false, msqVacuum, mg2, bUltraRelativistic);
	wallgo::ParticleSpecies lightQuark("quark", wallgo::EParticleType::FERMION, true, msqVacuum, mq2, bUltraRelativistic);

	manager.addParticle(topQuark);
	manager.addParticle(gluon);
	manager.addParticle(lightQuark);

	manager.setVariable("gs", gs);

	/* Where to load matrix elements from. If not specified, defaults to MatrixElements.txt in working dir. 
	This function returns false if the file is not found, in which case we abort here. */
	if (!manager.setMatrixElementFile("MatrixElements/MatrixElements_QCD.txt"))
	{
		std::cerr << "It looks like you may be running this example program from a nonstandard location.\n"
			"The matrix elements for this example are in MatrixElements/MatrixElements_QCD.txt which is hardcoded as a relative path for simplicity.\n"
			"Please run the example program inside the 'examples' directory.\n"
			"In your own applications you can use wallgo::CollisionManager::setMatrixElementFile() to specify the file location as you prefer."
			<< std::endl;
		return false;
	}

	return true;
}


int main() 
{

	// We use GSL for Monte Carlo integration. It needs to be initialized before use, with optional seed (default = 0)
    wallgo::gslWrapper::initializeRNG();

	// Can also set the seed at any later time:
	//wallgo::gslWrapper::setSeed(42);

    wallgo::CollisionManager manager;

	// Specify output directory (relative or absolute path). Defaults to current directory
	manager.setOutputDirectory("output");

	// Setup the particle content and specify matrix element file. If the setup fails we just abort
    if (!setupQCD(manager))
	{
		return 1;
	}

	// Tell the manager that it should print the symbolic matrix elements as they get parsed. 
	// This can be useful for debugging. By default this printing is disabled
	manager.setMatrixElementVerbosity(true);

	// Configure integrator. The defaults should be reasonably OK so you can only modify what you need.
	// Here we set everything manually to show how it's done
	wallgo::IntegrationOptions options;
	options.calls = 50000;
	options.maxTries = 50;
	options.maxIntegrationMomentum = 20;
	options.absoluteErrorGoal = 1e-8;
	options.relativeErrorGoal = 1e-1;
	options.bOptimizeUltrarelativistic = true;
	options.bVerbose = true;
	
	manager.configureIntegration(options);

	// Polynomial basis size. Using a trivially small N to make the example run fast
	const int basisSizeN = 3;

	manager.calculateCollisionIntegrals(basisSizeN, /*verbose*/true);


    wallgo::gslWrapper::clearRNG();
    
    return 0;
}