#ifndef COLLISIONINTEGRAL_H
#define COLLISIONINTEGRAL_H

#include <cmath>
#include <vector>

// Monte Carlo integration
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte_vegas.h>

#include "FourVector.h"
#include "CollElem.h"
#include "constants.h"
#include "PolynomialBasis.h"


void calculateAllCollisions();


// 2 -> 2 collision term integration. One particle is fixed as the "incoming" particle whose momentum is NOT integrated over. 
// This is always assumed to be first particle in collisionElements[i].particles.
// Momenta are denoted p1, p2 ; p3, p4.
// Assumes a 5D integral of form:
// int_0^infty p2^2/E2 dp2 p3^2/E3 dp3 int_0^(2pi) dphi2 dphi3 int_-1^1 dcosTheta2 dcosTheta3 Theta(E4) delta(P4^2 - m4^2) sum(|M|^2 P[ij -> mn])
// So 9D -> 5D reduction has been done analytically and this class calculates the rest.
class CollisionIntegral4 {

public:

    CollisionIntegral4(int polynomialBasisSize) : polynomialBasis(polynomialBasisSize) {
        //------ GSL initialization
        gsl_rng_env_setup();
        // Create a random number generator for the integration
        gslRNG = gsl_rng_alloc(gsl_rng_default);
    }

    ~CollisionIntegral4() {
        gsl_rng_free(gslRNG);
    }  

    void addCollisionElement(const CollElem<4> &collElem) { 
        collisionElements.push_back(collElem); 
        // TODO should check here that the collision element makes sense: has correct p1 particle etc
    }


    // ??
    void setIntegrationVariables(double p2, double phi2, double phi3, double cosTheta2, double cosTheta3) {
        integrationVariables.p2 = p2;
        integrationVariables.phi2 = phi2;
        integrationVariables.phi3 = phi3;
        integrationVariables.cosTheta2 = cosTheta2;
        integrationVariables.cosTheta3 = cosTheta3;
    }

    //----------------- Precalculate stuff for optimization

    // This also sets m, n, j, k internally, which is very important for the integrand function, so rename this
    void precalculateMomentum1(int m, int n, int j, int k) {
        gridIndices.m = m;
        gridIndices.n = n;
        gridIndices.j = j;
        gridIndices.k = k;

        rhoZ1 = polynomialBasis.rhoZGrid(j);
        rhoPar1 = polynomialBasis.rhoParGrid(k);
        pZ1 = polynomialBasis.rhoZ_to_pZ(rhoZ1);
        pPar1 = polynomialBasis.rhoPar_to_pPar(rhoPar1);
        TmTn_p1 = polynomialBasis.TmTn(m, n, rhoZ1, rhoPar1);
        p1 = std::sqrt(pZ1*pZ1 + pPar1*pPar1);
    }

    //----------------- End precalculations

    // Calculate the integral with Monte Carlo vegas. As always, mn = polynomial indices, jk = grid momentum indices
    // Returns { result, error }
    std::array<double, 2> evaluate(int m, int n, int j, int k, const std::array<double, 4> &massSquare);

    // TODO Would like a function fixFourMomenta(input_integration_variables) that applies delta functions and other kinematic constraints, 
    // then returns the external four-momenta. Right now its all done in calculateIntegrand()

    // Evaluate all 'collision elements' at input momenta, sum them and calculate the kinematic prefactor (including integration measure)
    // For now this takes particle masses as input. Those are also contained in CollElems but I don't have a simple way of extracting them from there
    // Very messy...
    // Specifically, this calculates the whole collision integrand as defined in eq. (A1) of 2204.13120 (linearized P), including the 1/(2N) prefactor.
    // NB: when comparing to output of Benoit's python code, note that he calculates 4pi * (A1), and that his tg_tg term had an error in the population factor.
    double calculateIntegrand(double p2, double phi2, double phi3, double cosTheta2, double cosTheta3, const std::array<double, 4> &massSquared);


    inline size_t getPolynomialBasisSize() const { return polynomialBasis.getBasisSize(); }

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

    Chebyshev polynomialBasis;

    //--------------------------

    // Basis polynomial indices m, n
    struct GridIndices {
        // Basis polynomial indices (Tbar_m, Ttilde_n)
        int m, n;
        // Grid momentum indices (j -> rho_z, k -> rho_par)
        int j, k;
    } gridIndices;


    // These will be pre-calculated before starting integration
    double rhoZ1, rhoPar1;
    double pZ1, pPar1;
    // Magnitude of p1 3-vector
    double p1;
    // Tm(rhoZ1)*Tn(rhoPar1)
    double TmTn_p1; 

    static constexpr size_t integralDimension = 5;
    gsl_rng* gslRNG;
};



#endif // header guard