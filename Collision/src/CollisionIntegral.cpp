
#include <iostream>
#include <assert.h>

#include "CollisionIntegral.h"
#include "CollElem.h"
#include "FourVector.h"
#include "ThreeVector.h"

#include "gslWrapper.h"
#include "ConfigParser.h"

// This calculates the full collision integral C[m,n; j,k]. NOTE: has to be thread safe!!
std::array<double, 2> CollisionIntegral4::evaluate(int m, int n, int j, int k)
{

    IntegrandParameters integrandParameters = initializeIntegrandParameters(m, n, j, k);

    // Integral dimensions (do NOT change this)
    constexpr size_t dim = 5;

    // Read integration options from config
    ConfigParser &config = ConfigParser::get();

    const double maxIntegrationMomentum = config.getDouble("Integration", "maxIntegrationMomentum");
    const size_t calls = config.getInt("Integration", "calls");
    const double relativeErrorGoal = std::fabs(config.getDouble("Integration", "relativeErrorGoal"));
    const double absoluteErrorGoal = std::fabs(config.getDouble("Integration", "absoluteErrorGoal"));
    const int maxTries = config.getInt("Integration", "maxTries");
    const bool bVerbose = config.getBool("Integration", "verbose");

    bOptimizeUltrarelativistic = config.getBool("Integration", "optimizeUltrarelativistic");

    // Define the integration limits for each variable: {p2, phi2, phi3, cosTheta2, cosTheta3}
    double integralLowerLimits[dim] = {0.0, 0.0, 0.0, -1., -1.};                                                  // Lower limits
    double integralUpperLimits[dim] = {maxIntegrationMomentum, 2.0 * constants::pi, 2.0 * constants::pi, 1., 1.}; // Upper limits

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
    gsl_monte_vegas_state *gslState = gsl_monte_vegas_alloc(dim);

    // Start with a short warmup run. This is good for importance sampling
    const size_t warmupCalls = 0.2 * calls;
    gsl_monte_vegas_integrate(&G, integralLowerLimits, integralUpperLimits, dim, warmupCalls, gslWrapper::rng, gslState, &mean, &error);

    // Lambda to check if we've reached the accuracy goal. This requires chisq / dof to be consistent with 1,
    // otherwise the error is not reliable
    auto hasConverged = [&gslState, &mean, &error, &relativeErrorGoal, &absoluteErrorGoal]()
    {
        bool bConverged = false;

        double chisq = gsl_monte_vegas_chisq(gslState); // the return value is actually chisq / dof
        if (std::fabs(chisq - 1.0) > 0.5)
        {
            // Error not reliable
        }
        // Handle case where integral is very close to 0
        else if (std::fabs(mean) < absoluteErrorGoal)
        {

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
    while (!hasConverged())
    {

        gsl_monte_vegas_integrate(&G, integralLowerLimits, integralUpperLimits, dim, calls, gslWrapper::rng, gslState, &mean, &error);

        currentTries++;
        if (currentTries >= maxTries)
        {
            if (bVerbose)
            {
                std::cerr << "Warning: Integration failed to reach accuracy goal. Result: " << mean << " +/- " << error << "\n";
            }
            break;
        }
    }

    gsl_monte_vegas_free(gslState);

    return std::array<double, 2>({mean, error});
}

void CollisionIntegral4::addCollisionElement(const CollElem<4> &elem)
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

std::vector<Kinematics> CollisionIntegral4::calculateKinematics(const CollElem<4> &collElem, double p1, double p2, const ThreeVector &p1Vec, const ThreeVector &p2Vec,
                                                                const ThreeVector &p3VecHat, double p1p2Dot, double p1p3HatDot, double p2p3HatDot)
{
    // Vacuum masses squared
    std::array<double, 4> massSquared;
    for (int i = 0; i < 4; ++i)
    {
        massSquared[i] = collElem.particles[i].getVacuumMassSquared();
    }

    // Energies: Since p3 is not fixed yet we only know E1, E2
    const double E1 = std::sqrt(p1 * p1 + massSquared[0]);
    const double E2 = std::sqrt(p2 * p2 + massSquared[1]);

    // Now def. function g(p3) = FV4(p3) - msq4 and express the delta function in
    // terms of roots of g(p3) and delta(p3 - p3root); p3 integral becomes trivial.
    // Will need to solve a quadratic equation; some helper variables for it:
    const double Q = massSquared[0] + massSquared[1] + massSquared[2] - massSquared[3];
    const double kappa = Q + 2.0 * (E1 * E2 - p1p2Dot);
    const double eps = 2.0 * (E1 + E2);
    const double delta = 2.0 * (p1p3HatDot + p2p3HatDot);

    // The eq is this. Could be used for failsafe checks, but so far haven't had any issues so commented out */
    /*
    auto funcG = [&](double p3) {
         double m3sq = massSquared[2];
         return kappa + delta*p3 - eps * sqrt(p3*p3 + m3sq);
    };
    */

    // Quadratic eq. A p3^2 + B p3 + C = 0, take positive root(s)
    const double A = delta * delta - eps * eps;
    const double B = 2.0 * kappa * delta;
    const double C = kappa * kappa - eps * eps * massSquared[2];
    // Roots of g(p3):
    const double discriminant = B * B - 4.0 * A * C;
    const double root1 = 0.5 * (-B - sqrt(discriminant)) / A;
    const double root2 = 0.5 * (-B + sqrt(discriminant)) / A;

    // Since p3 is supposed to be magnitude, pick only positive roots (p3 = 0 contributes nothing to the integral)
    const std::vector<double> roots{root1, root2};

    std::vector<Kinematics> kinematics;
    kinematics.reserve(2);

    for (double p3 : roots)
        if (p3 > 0)
        {
            const double E3 = std::sqrt(p3 * p3 + massSquared[2]);

            // E4 is fixed by 4-momentum conservation. There is a theta(E4) so we only accept E4 >= 0
            const double E4 = E1 + E2 - E3;
            if (E4 < 0)
                continue;

            Kinematics newKinematics;

            // Make four vectors
            newKinematics.FV1 = FourVector(E1, p1Vec);
            newKinematics.FV2 = FourVector(E2, p2Vec);
            newKinematics.FV3 = FourVector(E3, p3 * p3VecHat);

            newKinematics.FV4 = newKinematics.FV1 + newKinematics.FV2 - newKinematics.FV3;

            // g'(p3). Probably safe since we required p3 > 0, so E3 > 0:
            const double gDer = delta - eps * p3 / E3;

            newKinematics.prefactor = p2 * p3 * (p2 / (E2 + SMALL_NUMBER)) * (p3 / (E3 + SMALL_NUMBER)) / std::abs(gDer);

            kinematics.push_back(newKinematics);
        }

    return kinematics;
}

Kinematics CollisionIntegral4::calculateKinematics_ultrarelativistic(double p1, double p2, const ThreeVector &p1Vec, const ThreeVector &p2Vec, const ThreeVector &p3VecHat, double p1p2Dot, double p1p3HatDot, double p2p3HatDot)
{
    // Now E_i = p_i and the momentum-conserving delta function gives p3 = (p1*p2 - p1.p2 ) / (p1 + p2 - p3hat.(p1+p2))

    const double denom = p1 + p2 - p1p3HatDot - p2p3HatDot;
    const double p3 = (p1 * p2 - p1p2Dot) / denom;

    assert(p3 >= 0);

    Kinematics newKinematics;

    // Make four vectors
    newKinematics.FV1 = FourVector(p1, p1Vec);
    newKinematics.FV2 = FourVector(p2, p2Vec);
    newKinematics.FV3 = FourVector(p3, p3 * p3VecHat);
    
    newKinematics.FV4 = newKinematics.FV1 + newKinematics.FV2 - newKinematics.FV3;

    // g'(p3)
    const double gDer = -2 * denom;

    newKinematics.prefactor = p2*p3 / std::abs(gDer);

    return newKinematics;
}


double CollisionIntegral4::evaluateCollisionElement(CollElem<4> &collElem, const Kinematics &kinematics, const std::array<double, 4> &TmTn)
{
    // Need to set correct deltaF factors in collElem
    std::array<double, 4> deltaF;
    for (int i = 0; i < 4; ++i)
    {
        if (collElem.particles[i].isInEquilibrium())
        {
            deltaF[i] = 0.0;
        }
        else
        {
            deltaF[i] = TmTn[i];
        }
    }

    // Evaluate |M|^2 P[f] for this CollElem
    const std::array<FourVector, 4> fourMomenta({kinematics.FV1, kinematics.FV2, kinematics.FV3, kinematics.FV4});
    double res = collElem.evaluate(fourMomenta, deltaF);

    // Multiply by p2^2 / E2 * p3^2 / E3 |1/g'(p3)|
    res *= kinematics.prefactor;

    return res;
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

    // Create momentum 3-vectors
    ThreeVector p1Vec(pPar1, 0, pZ1);
    ThreeVector p2Vec(p2 * sinTheta2 * cosPhi2, p2 * sinTheta2 * sinPhi2, p2 * cosTheta2);

    // 'p3VecHat': like p3Vec, but normalized to 1. We will fix its magnitude using a delta(FV4^2 - msq4)
    const ThreeVector p3VecHat(sinTheta3 * cosPhi3, sinTheta3 * sinPhi3, cosTheta3);

    // Precalculate some dot products of 3-vectors that are same for all CollElems
    const double p1p2Dot = p1Vec * p2Vec;
    const double p1p3HatDot = p1Vec * p3VecHat;
    const double p2p3HatDot = p2Vec * p3VecHat;

    // Kinematics differs for each collision element since the masses are generally different.
    // But for CollElems with only ultrarelativistic (UR) particles the kinematic factors will always be the same

    // Handle these after doing optimized evaluation of the UR parts
    std::vector<CollElem<4>>* remainingCollElems = &collisionElements;

    if (bOptimizeUltrarelativistic && collisionElements_ultrarelativistic.size() > 0)
    {
        remainingCollElems = &collisionElements_nonUltrarelativistic;

        Kinematics kinematics = calculateKinematics_ultrarelativistic(p1, p2, p1Vec, p2Vec, p3VecHat, p1p2Dot, p1p3HatDot, p2p3HatDot);

        // Account for theta(E4). Note that in the non-UR case this is built-in to the kinematics computation
        if (kinematics.FV4.energy() >= 0)
        {
            // Calculate polynomial factors (which our method uses as replacement for deltaF): Tm(rhoZ) Tn(rhoPar)
            const std::array<double, 4> TmTn {
                integrandParameters.TmTn_p1,
                polynomialBasis.TmTn(m, n, kinematics.FV2),
                polynomialBasis.TmTn(m, n, kinematics.FV3),
                polynomialBasis.TmTn(m, n, kinematics.FV4)
            };
            
            for (CollElem<4> &collElem : collisionElements_ultrarelativistic)
            {
                fullIntegrand += evaluateCollisionElement(collElem, kinematics, TmTn);
            }
        }
    }

    // Non-ultrarelativistic computations. Here each CollElem has its own kinematics and the momentum delta-function needs a sum over p3 roots
    for (CollElem<4> &collElem : *remainingCollElems)
    {
        const std::vector<Kinematics> kinematicFactors = calculateKinematics(collElem, p1, p2, p1Vec, p2Vec, p3VecHat, p1p2Dot, p1p3HatDot, p2p3HatDot);

        // kinematicFactors only contains solutions with E4 > 0 so the theta(E4) is OK.
        for (const Kinematics &kinematics : kinematicFactors)
        {
            // Calculate polynomial factors (which our method uses as replacement for deltaF): Tm(rhoZ) Tn(rhoPar)
            const std::array<double, 4> TmTn {
                integrandParameters.TmTn_p1,
                polynomialBasis.TmTn(m, n, kinematics.FV2),
                polynomialBasis.TmTn(m, n, kinematics.FV3),
                polynomialBasis.TmTn(m, n, kinematics.FV4)
            };

            fullIntegrand += evaluateCollisionElement(collElem, kinematics, TmTn);
        }

    } // end collElem : collisionElements

    // Common numerical prefactor
    constexpr double PI = constants::pi;
    constexpr double pi2Pow58 = (2.0 * PI) * (2.0 * PI) * (2.0 * PI) * (2.0 * PI) * (2.0 * PI) * 8.0;
    return fullIntegrand / pi2Pow58;
}