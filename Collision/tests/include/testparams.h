#include "ParticleSpecies.h"
#include "CollisionIntegral.h"

namespace testparams {

    //**** Masses squared. These need to be in units of temperature, ie. (m/T)^2 **//
    // Thermal
    static double mq2 = 0.251327; // quark
    static double mg2 = 3.01593; // SU(3) gluon
    // Vacuum
    // TODO if needed
    static double msqVacuum = 0.0;

    static ParticleSpecies topQuark("top", EParticleType::FERMION, false, msqVacuum, mq2);
	static ParticleSpecies lightQuark("quark", EParticleType::FERMION, true, msqVacuum, mq2);
	static ParticleSpecies gluon("gluon", EParticleType::BOSON, true, msqVacuum, mg2);

    // Polynomial basis size
    static std::size_t basisSize = 20;

    // integration variables (I picked some random values)
    static double p2 = 5.3;
    static double phi2 = 2.7;
    static double phi3 = 4.1;
    static double cosTheta2 = -0.42;
    static double cosTheta3 = 0.83;
};