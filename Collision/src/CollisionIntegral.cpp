
#include "CollisionIntegral.h"
#include "CollElem.h"
#include <iostream>

// Monte Carlo integration
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte_vegas.h>

namespace gslWrapper {
     // Helpers for GSL integration routines. note that we cannot pass a member function by reference,
     // so we dodge this in the wrapper by passing a reference to the object whose function we want to integrate

     struct functionParams {
          int m, n;
          int j, k;
          std::array<double, 4> msq;
          // Pointer to the object whose member function we are integrating
          CollisionIntegral4* pointerToObject;
     };

     // pp should be of gslFunctionParams type
     inline double integrandWrapper(double* intVars, size_t dim, void* pp) {
          
          // GSL requires size_t as input: the logic is that dim should be used to check that intVars is correct size.
          // But I'm a badass and just do this:
          (void)dim;

          functionParams* params = static_cast<functionParams*>(pp);

          double p2 = intVars[0];
          double phi2 = intVars[1];
          double phi3 = intVars[2];
          double cosTheta2 = intVars[3];
          double cosTheta3 = intVars[4];
          return params->pointerToObject->calculateIntegrand(params->m, params->n, params->j, params->k, 
                                                                 p2, phi2, phi3, cosTheta2, cosTheta3, params->msq);
     } 
}


std::array<double, 2> CollisionIntegral4::evaluate(int m, int n, int j, int k, const std::array<double, 4> &massSquare) {

     // Integral dimensions
     const int dim = this->integralDimension;
     // Define the integration limits for each variable: {p2, phi2, phi3, cosTheta2, cosTheta3}
     double integralLowerLimits[dim] = {0.0, 0.0, 0.0, -1., -1.}; // Lower limits
     double integralUpperLimits[dim] = {maxIntegrationMomentum, 2.0*constants::pi, 2.0*constants::pi, 1., 1.}; // Upper limits

     // Construct parameter wrapper struct
     gslWrapper::functionParams paramWrap;
     paramWrap.m = m;
     paramWrap.n = n;
     paramWrap.j = j;
     paramWrap.k = k;
     paramWrap.msq = massSquare;
     paramWrap.pointerToObject = this;

     gsl_monte_function G;
     G.f = &gslWrapper::integrandWrapper;
     G.dim = dim;
     G.params = &paramWrap;

     // How many Monte Carlo iterations
     size_t calls = 50000;
     size_t warmupCalls = 0.1*calls;
     double mean = 0.0;
     double error = 0.0;

     // Unique Monte Carlo state for this integration. Other option would be to alloc this in the constructor 
     // and re-use the same state for all integrations (needs call to gsl_monte_vegas_init() for each new integral and would NOT be thread safe) 
     // but I did not notice any significant performance improvement over just allocating it here 
     gsl_monte_vegas_state* gslState = gsl_monte_vegas_alloc(dim);

     // Warmup?!?
     gsl_monte_vegas_integrate(&G, integralLowerLimits, integralUpperLimits, dim, warmupCalls, this->gslRNG, gslState, &mean, &error);
     // converge run??
     gsl_monte_vegas_integrate(&G, integralLowerLimits, integralUpperLimits, dim, calls, this->gslRNG, gslState, &mean, &error);
     
     gsl_monte_vegas_free(gslState);

     return std::array<double, 2>( {mean, error} );
}



// collision integral C[m,n; j,k]. mn = Chebyshev indices, jk = grid momentum indices. 
// Could optimize by pre-calculating p1 momenta etc
double CollisionIntegral4::calculateIntegrand(int m, int n, int j, int k, double p2, double phi2, double phi3, double cosTheta2, double cosTheta3, 
          const std::array<double, 4> &massSquared)
     {
     
     // Sines
     double sinTheta2 = std::sin(std::acos(cosTheta2));
     double sinTheta3 = std::sin(std::acos(cosTheta3));
     double sinPhi2 = std::sin(phi2);
     double sinPhi3 = std::sin(phi3);
     
     // Cosines
     double cosPhi2 = std::cos(phi2);
     double cosPhi3 = std::cos(phi3);

     // p1 3-vector and its magnitude
     double rhoZ1 = polynomialBasis.rhoZGrid(j);
     double rhoPar1 = polynomialBasis.rhoParGrid(k);

     double pZ1 = polynomialBasis.rhoZ_to_pZ(rhoZ1);
     double pPar1 = polynomialBasis.rhoPar_to_pPar(rhoPar1);
     double p1 = std::sqrt(pZ1*pZ1 + pPar1*pPar1);

     //printf("pZ1 %g pPar1 %g\n", pZ1, pPar1);
     
     
     // SLOPPY: Create 3-vectors, but I just use FourVectors with vanishing 0-component
     FourVector FV1dummy(0.0, pPar1, 0.0, pZ1);
     FourVector FV2dummy(0.0, p2*sinTheta2*cosPhi2, p2*sinTheta2*sinPhi2, p2*cosTheta2);
     // 'p3Hat': like p3, but normalized to 1. We will fix its magnitude using a delta(FV4^2 - msq4)
     FourVector FV3Hat(0.0, sinTheta3*cosPhi3, sinTheta3*sinPhi3, cosTheta3);

     // dot products of 3-vectors. Need minus sign here since I'm hacking this with 4-vectors with (+1 -1 -1 -1) metric
     double p1p2Dot = -1.0 * FV1dummy*FV2dummy;
     double p1p3HatDot = -1.0 * FV1dummy*FV3Hat;
     double p2p3HatDot = -1.0 * FV2dummy*FV3Hat;

     // Energies: Since p3 is not fixed yet we only know E1, E2 
     double E1 = std::sqrt(p1*p1 + massSquared[0]);
     double E2 = std::sqrt(p2*p2 + massSquared[1]);

     //------------------------------- TODO move this bit elsewhere
     
     // Now def. function g(p3) = FV4(p3) - msq4 and express the delta function in terms of roots of g(p3) and delta(p3 - p3root); p3 integral becomes trivial.
     // Will need to solve a quadratic equation; some helper variables for it:
     double Q = massSquared[0] + massSquared[1] + massSquared[2] - massSquared[3];
     double kappa = Q + 2.0 * (E1*E2 - p1p2Dot);
     double eps = 2.0 * (E1 + E2);
     double delta = 2.0 * (p1p3HatDot + p2p3HatDot);
     
     auto funcG = [&](double p3) {
          double m3sq = massSquared[2];
          return kappa + delta*p3 - eps * sqrt(p3*p3 + m3sq);
     };

     // Quadratic eq. A p3^2 + B p3 + C = 0, take positive root(s)
     double A = delta*delta - eps*eps;
     double B = 2.0 * kappa * delta;
     double C = kappa*kappa - eps*eps*massSquared[2];
     // Roots of g(p3):
     double discriminant = B*B - 4.0*A*C;
     double root1 = 0.5 * (-B - sqrt(discriminant)) / A;
     double root2 = 0.5 * (-B + sqrt(discriminant)) / A;

     // TODO:
     // 1) Check for possible singularities
     // 2) Check that the root(s) satisfy the original eq. with square root
     // 3) Is it possible to have 2 positive roots??

     //-------------------------------
     std::array<double, 2> rootp3({root1, root2});

     //printf("%ld %g %g %g %g\n", rootp3.size(), root1, root2, funcG(root1), funcG(root2));

     // TODO need way of calculating and assigning deltaFs, or Chebyshevs. (maybe even give this a pointer to funct that calculates deltaF for given FourVector?)
     double fullIntegrand = 0.0;

     // Now proceed to fix remaining 4-momenta
     for (double p3 : rootp3) if (p3 >= 0.0) {


          if (std::abs(funcG(p3)) > 1e-8) {
               std::cerr << "! Invalid root in CollisionIntegral4::calculateIntegrand \n";
          }

          double E3 = std::sqrt(p3*p3 + massSquared[2]);

          // Fix 4-momenta for real this time
          FourVector FV1 = FV1dummy;
          FV1[0] = E1;
          FourVector FV2 = FV2dummy;
          FV2[0] = E2;
          FourVector FV3 = p3*FV3Hat;
          FV3[0] = E3;

          // momentum conservation fixed P4 
          FourVector FV4 = FV1 + FV2 - FV3;

          if (FV4.energy() < 0.0) {
               std::cerr << "! Negative energy E4: " << FV4.energy() << "\n";
               continue;
          }
          // Check that P4 is on-shell // TODO

          //printf("p1 %g p2 %g p3 %g p4 %g\n", p1, p2, p3, FV4.norm3());

          // Now add all collision elements at these momenta
          double integrand = 0.0;
          for (CollElem<4> collElem : collisionElements) {

               // Fix deltaF's. In our spectral approach this means that we replace deltaF with 
               // Tm(rhoZ) Tn(rhoPar), where Tm, Tn are appropriate basis polynomials

               // TODO optimize. this is slow because the same momenta are re-calculated for each collision element

               std::array<FourVector, 4> fourMomenta({FV1, FV2, FV3, FV4});

               for (int i=0; i<4; ++i) {
                    if ( ! collElem.particles[i].isInEquilibrium()) {
                         collElem.particles[i].setDeltaF( polynomialBasis.TmTn(m, n, fourMomenta[i]) );
                    }
               }
               //printf("TmTn(p1) %g\n", polynomialBasis.TmTn(m, n, fourMomenta[0]));

               //printf("collision element %g\n", collElem.evaluate( fourMomenta ));

               integrand += collElem.evaluate( fourMomenta );
          } 

          // Kinematic prefactor
          double kinPrefac = 1.0;
          // Avoid spurious singularity at zero momenta, zero mass
          if (std::abs(massSquared[1]) < massSquaredLowerBound) kinPrefac *= p2;
          else kinPrefac *= p2*p2 / E2;

          if (std::abs(massSquared[2]) < massSquaredLowerBound) kinPrefac *= p3;
          else kinPrefac *= p3*p3 / E3;

          // additional factor from delta(g(p3))
          double gDer = 0.0;
          if (std::abs(massSquared[2]) < massSquaredLowerBound) gDer = delta - eps;
          else gDer = delta - eps * p3 / E3;

          integrand *= kinPrefac / std::abs(gDer);


          fullIntegrand += integrand;
     } // end p3 : rootp3
     

     double PI = constants::pi;
     double pi2Pow5 = (2.0*PI) * (2.0*PI) * (2.0*PI) * (2.0*PI) * (2.0*PI);
     return fullIntegrand / pi2Pow5 / 8.0;
}
