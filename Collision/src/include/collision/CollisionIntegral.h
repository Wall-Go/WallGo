#ifndef COLLISIONINTEGRAL_H
#define COLLISIONINTEGRAL_H

#include <cmath>
#include <vector>

#include "FourVector.h"
#include "CollElem.h"
#include "Common.h"
#include "PolynomialBasis.h"


// TEMPORARY. This is bound to the pybind module but does nothing ATM.  
inline void calculateAllCollisions() {}


/* This holds data for computing the "kinematic" factor in a collision integral. The kinematic factor is: 
    p2^2/E2 * p3^2/E3 * theta(E4) * delta(g(p3))
where the delta function enforces momentum conservation. Standard delta-trick expresses it as sum_i |1/g'(p3)| where we sum over roots of g(p3) = 0.
This struct describes one such root, and we only allow cases with p3 > 0, E4 >= 0
*/ 
struct KinematicFactor 
{
    double p3;
    // Storing this to avoid repeatedly taking sqrt. No need to store E4
    double E1, E2, E3;
    double prefactor; // This is p2^2/E2 * p3^2/E3 * |1 / g'(p3)|
};

/*
2 -> 2 collision term integration. One particle is fixed as the "incoming" particle whose momentum is NOT integrated over. 
This is always assumed to be first particle in collisionElements[i].particles.
Momenta are denoted p1, p2 ; p3, p4.
Assumes a 5D integral of form:
    int_0^infty p2^2/E2 dp2 p3^2/E3 dp3 int_0^(2pi) dphi2 dphi3 int_-1^1 dcosTheta2 dcosTheta3 Theta(E4) delta(P4^2 - m4^2) sum(|M|^2 P[ij -> mn])
So that 9D -> 5D reduction has been done analytically and this class calculates the rest.
*/
class CollisionIntegral4 {

public:

    // Struct to carry info about parameters other than the 5 integration variables. 
    // Optimization (precalculate p1 etc) + thread safety (since each thread can have its own copy of this struct)
    struct IntegrandParameters {
        // Basis polynomial indices (Tbar_m, Ttilde_n)
        int m, n;
        double rhoZ1, rhoPar1;
        double pZ1, pPar1;
        // Magnitude of p1 3-vector
        double p1;
        // Tm(rhoZ1)*Tn(rhoPar1)
        double TmTn_p1; 
    };


    CollisionIntegral4(std::size_t polynomialBasisSize) : polynomialBasis(polynomialBasisSize) {}

    ~CollisionIntegral4() {}  

    // 
    IntegrandParameters initializeIntegrandParameters(int m, int n, int j, int k) const {
        IntegrandParameters params;
        params.m = m;
        params.n = n;

        // Precalculate stuff related to the p1 momentum (optimization)
        params.rhoZ1 = polynomialBasis.rhoZGrid(j);
        params.rhoPar1 = polynomialBasis.rhoParGrid(k);
        params.pZ1 = polynomialBasis.rhoZ_to_pZ(params.rhoZ1);
        params.pPar1 = polynomialBasis.rhoPar_to_pPar(params.rhoPar1);
        params.TmTn_p1 = polynomialBasis.TmTn(m, n, params.rhoZ1, params.rhoPar1);
        params.p1 = std::sqrt(params.pZ1*params.pZ1 + params.pPar1*params.pPar1);
        return params;
    }

    /* Calculates the whole collision integrand as defined in eq. (A1) of 2204.13120 (linearized P). 
    Includes the 1/(2N) prefactor. Kinematics is solved (from delta functions) separately for each 
    CollElem in our collisionElements array. For ultrarelativistic CollElems we heavily optimize the kinematic part. */
    double calculateIntegrand(double p2, double phi2, double phi3, double cosTheta2, double cosTheta3, 
        const IntegrandParameters &integrandParameters);

    // Overload of the above (for testing)
    inline double calculateIntegrand(double p2, double phi2, double phi3, double cosTheta2, double cosTheta3, 
        int m, int n, int j, int k) {
            
        return calculateIntegrand(p2, phi2, phi3, cosTheta2, cosTheta3, initializeIntegrandParameters(m, n, j, k));
    }

    // Calculate the integral with Monte Carlo vegas. As always, mn = polynomial indices, jk = grid momentum indices
    // Returns { result, error }
    std::array<double, 2> evaluate(int m, int n, int j, int k);

    inline std::size_t getPolynomialBasisSize() const { return polynomialBasis.getBasisSize(); }

    void addCollisionElement(const CollElem<4>& elem);

private:

    // For avoiding 1/0
    const double SMALL_NUMBER = 1e-50;

    const Chebyshev polynomialBasis;

    // Should be no reason to disable this outside testing
    bool bOptimizeUltrarelativistic = true;

    /* Kinematic factor depends on masses in the collision element so in principle each element has its own kinematics. 
    We also use a delta-function trick to do delta(g(p3)) as a sum over roots of g(p3) = 0 so this is returns a vector.
    Inputs: magnitudes of p1, p2, dot products p1.p2, p1.p3Hat, p2.p3Hat. Here p3Hat is unit vector in direction of p3 (the direction is known, this solves magnitude). */
    std::vector<KinematicFactor> calculateKinematicFactor(const CollElem<4> &collElem, double p1, double p2, double p1p2Dot, double p1p3HatDot, double p2p3HatDot);

    /* Computes the kinematic factor for ultrarelativistic CollElems. This only depends on input momenta and not the CollElem itself. 
    In UR limit the momentum-conserving delta function only gives one solution for p3, so this one does not return an array. Used for optimization. */
    KinematicFactor calculateKinematicFactor_ultrarelativistic(double p1, double p2, double p1p2Dot, double p1p3HatDot, double p2p3HatDot);

    // 4-particle 'collision elements' that contribute to the process
    std::vector<CollElem<4>> collisionElements;

    /* We separate the collision elements into two subsets: ones with only ultrarelativistic (UR) external particles, and all others. 
    This allows optimizations related to the kinematic factor in UR terms. Not ideal to have both these and the full array though */
    std::vector<CollElem<4>> collisionElements_ultrarelativistic;
    std::vector<CollElem<4>> collisionElements_nonUltrarelativistic;
};



#endif // header guard