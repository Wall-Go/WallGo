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
    ParticleSpecies topQuark("top", EParticleType::FERMION, /*in-eq*/false, msqVacuum, mq2, bUltraRelativistic);
    ParticleSpecies gluon("gluon", EParticleType::BOSON, false, msqVacuum, mg2, bUltraRelativistic);
	ParticleSpecies lightQuark("quark", EParticleType::FERMION, true, msqVacuum, mq2, bUltraRelativistic);

	manager.addParticle(topQuark);
	manager.addParticle(gluon);
	manager.addParticle(lightQuark);

	manager.addCoupling(gs);
}


int main() 
{
    
	// basis size
	const int basisSizeN = 5;
    // config file, default name
    std::string configFileName = "CollisionDefaults.ini";

	// Load config
	ConfigParser& config = ConfigParser::get();

	if (config.load(configFileName)) {
		std::cout << "Read config:\n";
		config.printContents();
		std::cout << std::endl;
	} else {
		return 3;
	}

    gslWrapper::initializeRNG();

    CollisionManager manager;

    setupQCD(manager);

    manager.calculateCollisionIntegrals(basisSizeN);

    gslWrapper::clearRNG();
    
    return 0;
}