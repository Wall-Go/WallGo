#include "ParticleSpecies.h"
#include "CollisionIntegral.h"
#include "Collision.h"

#include <vector>

// ---- Configure test model (QCD)

class TestModel_QCD
{

public:
    TestModel_QCD() : collision(basisSizeN)
    {
        particles.push_back(ParticleSpecies("top", EParticleType::FERMION, /*ineq*/false, topMsqVacuum, mq2, true));
        particles.push_back(ParticleSpecies("gluon", EParticleType::BOSON, /*ineq*/false, gluonMsqVacuum, mg2, true));
        particles.push_back(ParticleSpecies("lightQuark", EParticleType::FERMION, /*ineq*/true, 0.0, mq2, true));

        collision.addParticle(ParticleSpecies("top", EParticleType::FERMION, /*ineq*/false, topMsqVacuum, mq2, true));
        collision.addParticle(ParticleSpecies("gluon", EParticleType::BOSON, /*ineq*/false, gluonMsqVacuum, mg2, true));
        collision.addParticle(ParticleSpecies("lightQuark", EParticleType::FERMION, /*ineq*/true, 0.0, mq2, true));

        collision.addCoupling(gs)

    };

    Collision collision;

private:

    const uint basisSizeN = 11;


    // QCD coupling
    const double gs = 1.2279920495357861;

    //**** Masses squared. These need to be in units of temperature, ie. (m/T)^2 **//
    // Thermal
    const double mq2 = 0.251327; // quark
    const double mg2 = 3.01593; // SU(3) gluon
    // Vacuum
    const double topMsqVacuum = 0.0;
    const double gluonMsqVacuum = 0.0;

    std::vector<ParticleSpecies> particles;

};

class TestParams {

    //**** Masses squared. These need to be in units of temperature, ie. (m/T)^2 **//
    // Thermal
    const double mq2 = 0.251327; // quark
    const double mg2 = 3.01593; // SU(3) gluon
    // Vacuum
    // TODO if needed
    const double msqVacuum = 0.0;
    const bool bUltraRelativistic = true;


    const ParticleSpecies topQuark;
    const ParticleSpecies lightQuark;
	const ParticleSpecies gluon;
    
    TestParams() : topQuark("top", EParticleType::FERMION, false, msqVacuum, mq2, bUltraRelativistic),
                    lightQuark("quark", EParticleType::FERMION, true, msqVacuum, mq2, bUltraRelativistic),
                    gluon("gluon", EParticleType::BOSON, true, msqVacuum, mg2, bUltraRelativistic) {}


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