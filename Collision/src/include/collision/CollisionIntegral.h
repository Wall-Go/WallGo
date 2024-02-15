#ifndef COLLISIONINTEGRAL_H
#define COLLISIONINTEGRAL_H

#include <cmath>
#include <vector>

#include "FourVector.h"
#include "ThreeVector.h"
#include "CollElem.h"
#include "Common.h"
#include "PolynomialBasis.h"

/* This holds data for computing the "kinematic" factor in a collision integral. The kinematic factor is: 
    p2^2/E2 * p3^2/E3 * theta(E4) * delta(g(p3))
where the delta function enforces momentum conservation. Standard delta-trick expresses it as sum_i |1/g'(p3)| where we sum over roots of g(p3) = 0.
This struct describes one such root, and we only allow cases with p3 > 0, E4 >= 0.
*/ 
struct Kinematics 
{
    FourVector FV1, FV2, FV3, FV4;
    double prefactor; // This is p2^2/E2 * p3^2/E3 * |1 / g'(p3)|
};

struct IntegrationOptions
{
    double maxIntegrationMomentum;
    // How many Monte Carlo calls between convergence checks 
    size_t calls;
    double relativeErrorGoal;
    double absoluteErrorGoal;
    double maxTries;
    bool bVerbose;
    bool bOptimizeUltrarelativistic;
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
    std::array<double, 2> integrate(int m, int n, int j, int k, const IntegrationOptions& options);

    inline std::size_t getPolynomialBasisSize() const { return polynomialBasis.getBasisSize(); }

    void addCollisionElement(const CollElem<4>& elem);

    /* Enables faster computation of kinematic factors for ultrarelativistic collision elements. Should be no reason to disable this outside testing */
    bool bOptimizeUltrarelativistic = true;

private:

    // For avoiding 1/0
    const double SMALL_NUMBER = 1e-50;

    const Chebyshev polynomialBasis;

    /* Kinematic factor depends on masses in the collision element so in principle each element has its own kinematics. 
    We also use a delta-function trick to do delta(g(p3)) as a sum over roots of g(p3) = 0 so this is returns a vector.
    This takes a bunch of vectors, their magnitudes and dot products as input, there is some redundancy but we're trying to avoid having to compute them many times. */
    std::vector<Kinematics> calculateKinematics(const CollElem<4> &collElem, double p1, double p2, 
        const ThreeVector& p1Vec, const ThreeVector& p2Vec, const ThreeVector& p3VecHat, double p1p2Dot, double p1p3HatDot, double p2p3HatDot);

    /* Optimized computation of the kinematic factor for ultrarelativistic CollElems. This only depends on input momenta and not the CollElem itself. 
    In UR limit the momentum-conserving delta function gives only one solution for p3, so this one does not return an array. */
    Kinematics calculateKinematics_ultrarelativistic(double p1, double p2, 
        const ThreeVector& p1Vec, const ThreeVector& p2Vec, const ThreeVector& p3VecHat, double p1p2Dot, double p1p3HatDot, double p2p3HatDot);


    /* Evaluate |M^2|/N * P[TmTn] * (kinematics.prefactor). TmTn are the polynomial factors evaluated at each momenta. TODO make this so that collElem can be const...? */
    double evaluateCollisionElement(CollElem<4> &collElem, const Kinematics& kinematics, const std::array<double, 4>& TmTn);

    // 4-particle 'collision elements' that contribute to the process
    std::vector<CollElem<4>> collisionElements;

    /* We separate the collision elements into two subsets: ones with only ultrarelativistic (UR) external particles, and all others. 
    This allows optimizations related to the kinematic factor in UR terms. Not ideal to have both these and the full array though */
    std::vector<CollElem<4>> collisionElements_ultrarelativistic;
    std::vector<CollElem<4>> collisionElements_nonUltrarelativistic;
};



#endif // header guard