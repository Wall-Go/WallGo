#ifndef COLLISIONINTEGRAL_H
#define COLLISIONINTEGRAL_H

#include <cmath>
#include <vector>

#include "FourVector.h"
#include "CollElem.h"
#include "constants.h"
#include "PolynomialBasis.h"


// TEMPORARY. This is bound to the pybind module but does nothing ATM.  
inline void calculateAllCollisions() {}


// 2 -> 2 collision term integration. One particle is fixed as the "incoming" particle whose momentum is NOT integrated over. 
// This is always assumed to be first particle in collisionElements[i].particles.
// Momenta are denoted p1, p2 ; p3, p4.
// Assumes a 5D integral of form:
// int_0^infty p2^2/E2 dp2 p3^2/E3 dp3 int_0^(2pi) dphi2 dphi3 int_-1^1 dcosTheta2 dcosTheta3 Theta(E4) delta(P4^2 - m4^2) sum(|M|^2 P[ij -> mn])
// So that 9D -> 5D reduction has been done analytically and this class calculates the rest.
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

    void addCollisionElement(const CollElem<4> &collElem) { 
        collisionElements.push_back(collElem); 
        // TODO should check here that the collision element makes sense: has correct p1 particle etc
    }


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

    /* Evaluate all 'collision elements' at input momenta, sum them 
    and calculate the kinematic prefactor (including the integration measure)
    Specifically, this calculates the whole collision integrand as defined in eq. (A1) of 2204.13120 (linearized P). 
    Includes the 1/(2N) prefactor.
    NB: when comparing to output of Benoit's python code, note that he calculates 4pi * (A1), 
    and that his tg_tg term had an error in the population factor. */
    double calculateIntegrand(double p2, double phi2, double phi3, double cosTheta2, double cosTheta3, 
        const IntegrandParameters &integrandParameters) const;

    // Overload of the above (for testing)
    inline double calculateIntegrand(double p2, double phi2, double phi3, double cosTheta2, double cosTheta3, 
        int m, int n, int j, int k) const {
            
        return calculateIntegrand(p2, phi2, phi3, cosTheta2, cosTheta3, initializeIntegrandParameters(m, n, j, k));
    }

    // Calculate the integral with Monte Carlo vegas. As always, mn = polynomial indices, jk = grid momentum indices
    // Returns { result, error }
    std::array<double, 2> evaluate(int m, int n, int j, int k);

    inline std::size_t getPolynomialBasisSize() const { return polynomialBasis.getBasisSize(); }

    // 4-particle 'collision elements' that contribute to the process
    std::vector<CollElem<4>> collisionElements;

private:

    // upper limit on p2 integration
    const double maxIntegrationMomentum = 20.0; 

    // Masses smaller than this are set to 0 when computing the kinematic prefactor. This is to avoid spurious singularities at small momenta
    const double massSquaredLowerBound = 1e-14;

    const Chebyshev polynomialBasis;
};



#endif // header guard