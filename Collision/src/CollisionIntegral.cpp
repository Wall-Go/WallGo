
#include <iostream>

#include "CollisionIntegral.h"
#include "CollElem.h"

#include "gslWrapper.h"
#include "ConfigParser.h"

// This calculates the full collision integral C[m,n; j,k]. NOTE: has to be thread safe!!
std::array<double, 2> CollisionIntegral4::evaluate(int m, int n, int j, int k) {

     IntegrandParameters integrandParameters = initializeIntegrandParameters(m, n, j, k); 

     // Integral dimensions (do NOT change this)
     constexpr size_t dim = 5;

     // Read integration options from config
     ConfigParser& config = ConfigParser::get();

     const double maxIntegrationMomentum = config.getDouble("Integration", "maxIntegrationMomentum");
     const size_t calls = config.getInt("Integration", "calls");
     const double relativeErrorGoal = std::fabs( config.getDouble("Integration", "relativeErrorGoal") );
     const double absoluteErrorGoal = std::fabs( config.getDouble("Integration", "absoluteErrorGoal") );
     const int maxTries = config.getInt("Integration", "maxTries");
     const bool bVerbose = config.getBool("Integration", "verbose");

     // Define the integration limits for each variable: {p2, phi2, phi3, cosTheta2, cosTheta3}
     double integralLowerLimits[dim] = {0.0, 0.0, 0.0, -1., -1.}; // Lower limits
     double integralUpperLimits[dim] = {maxIntegrationMomentum, 2.0*constants::pi, 2.0*constants::pi, 1., 1.}; // Upper limits

     // Construct parameter wrapper struct
     gslWrapper::gslFunctionParams gslWrapper;
     gslWrapper.pointerToObject = this;
     gslWrapper.integrandParameters = integrandParameters;

     gsl_monte_function G;
     G.f = &gslWrapper::integrandWrapper;
     G.dim = dim;
     G.params = &gslWrapper;

     // Options for Vegas Monte Carlo integration. https://www.gnu.org/software/gsl/doc/html/montecarlo.html#vegas
     // By default each call to gsl_monte_vegas_integrate will perform 5 iterations of the algorithm. 
     // This could be changed with a new gsl_monte_vegas_params struct and passing that to gsl_monte_vegas_params_set().
     // Here we use default params struct and just set the number of calls
  
     double mean = 0.0;
     double error = 0.0;

     // Unique Monte Carlo state for this integration. NB: keep this here: thread safety
     gsl_monte_vegas_state* gslState = gsl_monte_vegas_alloc(dim);

     // Start with a short warmup run. This is good for importance sampling 
     const size_t warmupCalls = 0.2*calls;
     gsl_monte_vegas_integrate(&G, integralLowerLimits, integralUpperLimits, dim, warmupCalls, gslWrapper::rng, gslState, &mean, &error);
     

     // Lambda to check if we've reached the accuracy goal. This requires chisq / dof to be consistent with 1, 
     // otherwise the error is not reliable
     auto hasConverged = [&gslState, &mean, &error, &relativeErrorGoal, &absoluteErrorGoal]() {

          bool bConverged = false;

          double chisq = gsl_monte_vegas_chisq(gslState); // the return value is actually chisq / dof
          if (std::fabs(chisq - 1.0) > 0.5) {
               // Error not reliable
          }
          // Handle case where integral is very close to 0
          else if (std::fabs(mean) < absoluteErrorGoal) {
               
               bConverged = true;
          } 
          // Else: "standard" case
          else if (std::fabs(error / mean) < relativeErrorGoal)
          {
               bConverged = true;
          }
          return bConverged;
     };


     int currentTries = 0; 
     while (!hasConverged()) {
     
          gsl_monte_vegas_integrate(&G, integralLowerLimits, integralUpperLimits, dim, calls, gslWrapper::rng, gslState, &mean, &error);

          currentTries++;
          if (currentTries >= maxTries) {
               if (bVerbose) {
                    std::cerr << "Warning: Integration failed to reach accuracy goal. Result: " << mean << " +/- " << error << "\n";
               }
               break;
          }
     }

     gsl_monte_vegas_free(gslState);

     return std::array<double, 2>( {mean, error} );
}

void CollisionIntegral4::addCollisionElement(const CollElem<4>& elem)
{
     // TODO should check here that the collision element makes sense: has correct p1 particle etc

     collisionElements.push_back(elem);
     if (elem.isUltrarelativistic())
     {
          collisionElements_ultrarelativistic.push_back(elem);
     }
     else
     {
          collisionElements_nonUltrarelativistic.push_back(elem);
     }

}


std::vector<KinematicFactor> CollisionIntegral4::calculateKinematicFactor(const CollElem<4>& collElem, double p1, double p2, double p1p2Dot, double p1p3HatDot, double p2p3HatDot)
{
     // Vacuum masses squared
     std::array<double, 4> massSquared;
     for (int i=0; i<4; ++i) {
          massSquared[i] = collElem.particles[i].getVacuumMassSquared();
     }

     // Energies: Since p3 is not fixed yet we only know E1, E2
     double E1 = std::sqrt(p1*p1 + massSquared[0]);
     double E2 = std::sqrt(p2*p2 + massSquared[1]);

     // Now def. function g(p3) = FV4(p3) - msq4 and express the delta function in
     // terms of roots of g(p3) and delta(p3 - p3root); p3 integral becomes trivial.
     // Will need to solve a quadratic equation; some helper variables for it:
     double Q = massSquared[0] + massSquared[1] + massSquared[2] - massSquared[3];
     double kappa = Q + 2.0 * (E1*E2 - p1p2Dot);
     double eps = 2.0 * (E1 + E2);
     double delta = 2.0 * (p1p3HatDot + p2p3HatDot);

     // The eq is this. Could be used for failsafe checks, but so far haven't had any issues so commented out */
     /*
     auto funcG = [&](double p3) {
          double m3sq = massSquared[2];
          return kappa + delta*p3 - eps * sqrt(p3*p3 + m3sq);
     };
     */

     // Quadratic eq. A p3^2 + B p3 + C = 0, take positive root(s)
     double A = delta*delta - eps*eps;
     double B = 2.0 * kappa * delta;
     double C = kappa*kappa - eps*eps*massSquared[2];
     // Roots of g(p3):
     double discriminant = B*B - 4.0*A*C;
     double root1 = 0.5 * (-B - sqrt(discriminant)) / A;
     double root2 = 0.5 * (-B + sqrt(discriminant)) / A;

     // Since p3 is supposed to be magnitude, pick only positive roots (p3 = 0 contributes nothing)
     const std::vector<double> roots{ root1, root2 };

     std::vector<KinematicFactor> factors;
     factors.reserve(2);

     for (double p3 : roots) if (p3 > 0)
     {
          const double E3 = std::sqrt(p3*p3 + massSquared[2]);

          // E4 is fixed by 4-momentum conservation. There is a theta(E4) so we only accept E4 >= 0
          const double E4 = E1 + E2 - E3;

          if (E4 < 0) continue;

          KinematicFactor newFactor;
          newFactor.p3 = p3;
          newFactor.E1 = E1;
          newFactor.E2 = E2;
          newFactor.E3 = E3;

          // g'(p3). Probably safe since we required p3 > 0, so E3 > 0:
          const double gDer = delta - eps * p3 / E3;

          newFactor.prefactor = p2*p3 * (p2/(E2 + SMALL_NUMBER)) * (p3/(E3 + SMALL_NUMBER)) / std::abs(gDer);

          factors.push_back(newFactor);
     }

     // TODO:
     // 1) Check for possible singularities
     // 2) Check that the root(s) satisfy the original eq. with square root
     // 3) Is it even possible to have 2 positive roots??


    return factors;
}


double CollisionIntegral4::calculateIntegrand(double p2, double phi2, double phi3, double cosTheta2, double cosTheta3, 
        const IntegrandParameters &integrandParameters)
{
     
     double fullIntegrand = 0.0;

     // Variables from the input structure, redefine locally here for convenience
     const int m = integrandParameters.m;
     const int n = integrandParameters.n;

     const double p1 = integrandParameters.p1;
     const double pZ1 = integrandParameters.pZ1;
     const double pPar1 = integrandParameters.pPar1;

     // Sines
     const double sinTheta2 = std::sin(std::acos(cosTheta2));
     const double sinTheta3 = std::sin(std::acos(cosTheta3));
     const double sinPhi2 = std::sin(phi2);
     const double sinPhi3 = std::sin(phi3);
     
     // Cosines
     const double cosPhi2 = std::cos(phi2);
     const double cosPhi3 = std::cos(phi3);

     
     // SLOPPY: Create 3-vectors, but I just use FourVectors with vanishing 0-component
     FourVector FV1dummy(0.0, pPar1, 0.0, pZ1);
     FourVector FV2dummy(0.0, p2*sinTheta2*cosPhi2, p2*sinTheta2*sinPhi2, p2*cosTheta2);
     // 'p3Hat': like p3, but normalized to 1. We will fix its magnitude using a delta(FV4^2 - msq4)
     const FourVector FV3Hat(0.0, sinTheta3*cosPhi3, sinTheta3*sinPhi3, cosTheta3);

     // dot products of 3-vectors. Need minus sign here since I'm hacking this with 4-vectors with (+1 -1 -1 -1) metric
     const double p1p2Dot = -1.0 * FV1dummy*FV2dummy;
     const double p1p3HatDot = -1.0 * FV1dummy*FV3Hat;
     const double p2p3HatDot = -1.0 * FV2dummy*FV3Hat;

     // Kinematics differs for each collision process since the masses are generally different
     // TODO optimize so that if everything is ultrarelativistic, calculate kinematic factors only once
     for (CollElem<4> &collElem : collisionElements) {

          const std::vector<KinematicFactor> kinematicFactors = calculateKinematicFactor(collElem, p1, p2, p1p2Dot, p1p3HatDot, p2p3HatDot);

          // Now proceed to fix remaining 4-momenta
          for (const KinematicFactor& kinFactor : kinematicFactors)
          {

               // Fix 4-momenta for real this time
               FourVector FV1 = FV1dummy;
               FV1[0] = kinFactor.E1;
               FourVector FV2 = FV2dummy;
               FV2[0] = kinFactor.E2;
               FourVector FV3 = kinFactor.p3*FV3Hat;
               FV3[0] = kinFactor.E3;

               // momentum conservation fixes P4 
               const FourVector FV4 = FV1 + FV2 - FV3;

               // Calculate deltaF's for all momenta.
               // In our spectral approach this means that we replace deltaF with 
               // Tm(rhoZ) Tn(rhoPar), where Tm, Tn are appropriate basis polynomials
               const double TmTn_p1 = integrandParameters.TmTn_p1;
               const double TmTn_p2 = polynomialBasis.TmTn(m, n, FV2);
               const double TmTn_p3 = polynomialBasis.TmTn(m, n, FV3);
               const double TmTn_p4 = polynomialBasis.TmTn(m, n, FV4);
               const std::array<double, 4> TmTn_p({TmTn_p1, TmTn_p2, TmTn_p3, TmTn_p4});

               std::array<double, 4> deltaF;
                for (int i=0; i<4; ++i) {
                    if (collElem.particles[i].isInEquilibrium()) {
                         // particle species in equilibrium, so deltaF = 0
                         deltaF[i] = 0.0;
                    } else {
                         deltaF[i] = TmTn_p[i];
                    }
                }

               // Evaluate |M|^2 P[f] for this CollElem
               const std::array<FourVector, 4> fourMomenta({FV1, FV2, FV3, FV4});
               double integrand = collElem.evaluate( fourMomenta, deltaF );

               // Multiply by p2^2 / E2 * p3^2 / E3 |1/g'(p3)|
               integrand *= kinFactor.prefactor;

               fullIntegrand += integrand;
          } // end p3 : rootp3

     } // end collElem : collisionElements
     
     // Common numerical prefactor
     constexpr double PI = constants::pi;
     constexpr double pi2Pow58 = (2.0*PI) * (2.0*PI) * (2.0*PI) * (2.0*PI) * (2.0*PI) * 8.0;
     return fullIntegrand / pi2Pow58;
}