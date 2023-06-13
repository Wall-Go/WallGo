#ifndef COLLISIONINTEGRAL_H
#define COLLISIONINTEGRAL_H

#include <cmath>
#include <vector>

#include "FourVector.h"
#include "CollElem.h"

void calculateAllCollisions();


// 2 -> 2 collision term integration. One particle is fixed as the "incoming" particle whose momentum is NOT integrated over. 
// This is always assumed to be first particle in collisionElements[i].particles.
// Momenta are denoted p1, p2 ; p3, p4.
// Assumes a 5D integral of form:
// int_0^infty p2^2/E2 dp2 p3^2/E3 dp3 int_0^(2pi) dphi2 dphi3 int_-1^1 dcosTheta2 dcosTheta3 Theta(E4) delta(P4^2 - m4^2) sum(|M|^2 P[ij -> mn])
// So 9D -> 5D reduction has been done analytically and this class calculates the rest.
class CollisionIntegral4 {

public:

    void addCollisionElement(const CollElem<4> &collElem) { 
        collisionElements.push_back(collElem); 
        // TODO should check here that the collision element makes sense: has correct p1 particle etc
    }


    void setIntegrationVariables(double p2, double phi2, double phi3, double cosTheta2, double cosTheta3) {
        integrationVariables.p2 = p2;
        integrationVariables.phi2 = phi2;
        integrationVariables.phi3 = phi3;
        integrationVariables.cosTheta2 = cosTheta2;
        integrationVariables.cosTheta3 = cosTheta3;
    }

    std::array<FourVector, 4> fixFourMomenta();

    // Everything in the integral except for matrix element and population factor (TODO constant prefactors here or nay?)
    double kinematicPrefactor(const FourVector &p1, const FourVector &p2, const FourVector &p3, const FourVector &p4);

    // Evaluate all 'collision elements' at input momenta, sum them and calculate the kinematic prefactor (including integration measure)
    // For now this takes particle masses as input. Those are also contained in CollElems but I don't have a simple way of extracting them from there
    double calculateIntegrand(const std::array<double, 3> &p1Vec, double p2, double phi2, double phi3, double cosTheta2, double cosTheta3, const std::array<double, 4> &massSquared);

private:

    // dunno if need this
    struct IntegrationVariables {
        double p2; // magnitude of p2 3-momentum
        double phi2, phi3;
        double cosTheta2, cosTheta3;       
    } integrationVariables;

    // upper limit on p2 integration
    double maxIntegrationMomentum = 20.0; 
    // 4-particle 'collision elements' that contribute to the process
    std::vector<CollElem<4>> collisionElements;

    // Masses smaller than this are set to 0 when computing the kinematic prefactor. This is to avoid spurious singularities at small momenta
    double massSquaredLowerBound = 1e-14;
    // TODO this should be elsewhere since others may need it too
    const double PI = 3.141592653589793238463;
};



#endif // header guard