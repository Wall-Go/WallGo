#include "ParticleSpecies.h"
#include "CollisionIntegral.h"


struct TestParams {

    //**** Masses squared. These need to be in units of temperature, ie. (m/T)^2 **//
    // Thermal
    const double mq2 = 0.251327; // quark
    const double mg2 = 3.01593; // SU(3) gluon
    // Vacuum
    // TODO if needed
    const double msqVacuum = 0.0;

    const ParticleSpecies topQuark;
    const ParticleSpecies lightQuark;
	const ParticleSpecies gluon;

    TestParams() : topQuark("top", EParticleType::FERMION, false, msqVacuum, mq2),
                    lightQuark("quark", EParticleType::FERMION, true, msqVacuum, mq2),
                    gluon("gluon", EParticleType::BOSON, true, msqVacuum, mg2) {}


    // Polynomial basis size
    const std::size_t basisSize = 20;

    // integration variables (I picked some random values)
    const double p2 = 5.3;
    const double phi2 = 2.7;
    const double phi3 = 4.1;
    const double cosTheta2 = -0.42;
    const double cosTheta3 = 0.83;
};

// 'singleton' implementation of TestParams struct
inline TestParams& getTestParams() {
    static TestParams testParams;
    return testParams;
}