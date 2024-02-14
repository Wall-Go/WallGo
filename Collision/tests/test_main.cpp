#include <gtest/gtest.h>
#include "ParticleSpecies.h"
#include "Collision.h"



// ---- Configure test model (QCD)
Collision setupQCD(uint basisSize)
{
    const double gs = 1.2279920495357861;

    //**** Masses squared. These need to be in units of temperature, ie. (m/T)^2 **//
	// Thermal
	const double mq2 = 0.251327; // quark
	const double mg2 = 3.01593; // SU(3) gluon
	// Vacuum, @TODO if needed
	const double msqVacuum = 0.0;

	const bool bUltraRelativistic = true;

	// For now take just the top to be out-of-eq
    ParticleSpecies topQuark("top", EParticleType::FERMION, false, msqVacuum, mq2, bUltraRelativistic);
    ParticleSpecies gluon("gluon", EParticleType::BOSON, false, msqVacuum, mg2, bUltraRelativistic);
	ParticleSpecies lightQuark("quark", EParticleType::FERMION, true, msqVacuum, mq2, bUltraRelativistic);

	// Main control object
	Collision collision(N);
	collision.addParticle(topQuark);
	collision.addParticle(gluon);
	collision.addParticle(lightQuark);

	collision.addCoupling(gs);

	collision.calculateCollisionIntegrals();
}




int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
