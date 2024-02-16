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

#include "WallGoCollision/WallGoCollision.h"

void setupQCD(CollisionManager& manager) {

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
    ParticleSpecies topQuark("top", EParticleType::FERMION, /*inEquilibrium*/false, msqVacuum, mq2, bUltraRelativistic);
    ParticleSpecies gluon("gluon", EParticleType::BOSON, false, msqVacuum, mg2, bUltraRelativistic);
	ParticleSpecies lightQuark("quark", EParticleType::FERMION, true, msqVacuum, mq2, bUltraRelativistic);

	manager.addParticle(topQuark);
	manager.addParticle(gluon);
	manager.addParticle(lightQuark);

	manager.addCoupling(gs);

	// Where to load matrix elements from. If not specified, defaults to MatrixElements.txt in working dir
	manager.setMatrixElementFile("MatrixElements/MatrixElements_QCD.txt");
}


int main() 
{

    gslWrapper::initializeRNG();

    CollisionManager manager;

	// Specify output directory (relative or absolute path). Defaults to current directory
	manager.setOutputDirectory("output");

    setupQCD(manager);

	// Configure integrator. The defaults should be reasonably OK so you can only modify what you need.
	// Here we set everything manually to show how it's done
	IntegrationOptions options;
	options.calls = 50000;
	options.maxTries = 50;
	options.maxIntegrationMomentum = 20;
	options.absoluteErrorGoal = 1e-8;
	options.relativeErrorGoal = 1e-1;
	options.bOptimizeUltrarelativistic = true;
	options.bVerbose = true;
	
	manager.configureIntegration(options);

	// Polynomial basis size
	const int basisSizeN = 5;

    manager.calculateCollisionIntegrals(basisSizeN, /*verbose*/true);

    gslWrapper::clearRNG();
    
    return 0;
}