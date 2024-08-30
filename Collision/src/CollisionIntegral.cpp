
#include <iostream>
#include <cassert>
#include <chrono>

#include "EnvironmentMacros.h"
#include "CollisionIntegral.h"
#include "CollisionElement.h"
#include "FourVector.h"
#include "ThreeVector.h"
#include "ModelChangeContext.h"
#include "PolynomialBasis.h"

#include "gslWrapper.h"

#if WITH_OMP
    #include <omp.h>
#endif

namespace wallgo
{

IntegrationResult CollisionIntegral4::integrate(const GridPoint& gridPoint, const IntegrationOptions& options)
{
    assert(isValidGridPoint(gridPoint));
    
    IntegrandParameters integrandParameters = initializeIntegrandParameters(gridPoint);

    // Integral dimensions
    constexpr size_t dim = 5;

    // Options for Vegas Monte Carlo integration. https://www.gnu.org/software/gsl/doc/html/montecarlo.html#vegas
    // By default each call to gsl_monte_vegas_integrate will perform 5 iterations of the algorithm.
    // This could be changed with a new gsl_monte_vegas_params struct and passing that to gsl_monte_vegas_params_set().
    // Here we use default params struct and just set the number of calls

    const double maxIntegrationMomentum = options.maxIntegrationMomentum;
    const size_t calls = options.calls;
    const double relativeErrorGoal = options.relativeErrorGoal;
    const double absoluteErrorGoal = options.absoluteErrorGoal;
    const int maxTries = options.maxTries;

    bOptimizeUltrarelativistic = options.bOptimizeUltrarelativistic;

    // Define the integration limits for each variable: {p2, phi2, phi3, cosTheta2, cosTheta3}
    double integralLowerLimits[dim] = { 0.0, 0.0, 0.0, -1., -1. };
    double integralUpperLimits[dim] = { maxIntegrationMomentum, 2.0 * constants::pi, 2.0 * constants::pi, 1., 1. };

    // Construct parameter wrapper struct
    gslWrapper::gslFunctionParams gslWrapper;
    gslWrapper.pointerToObject = this;
    gslWrapper.integrandParameters = integrandParameters;

    gsl_monte_function gslFunction;
    gslFunction.f = &gslWrapper::integrandWrapper;
    gslFunction.dim = dim;
    gslFunction.params = &gslWrapper;

    double mean = 0.0;
    double error = 0.0;

    // Unique Monte Carlo state for this integration. NB: keep this here: thread safety
    gsl_monte_vegas_state* gslState = gsl_monte_vegas_alloc(dim);

    // Start with a short warmup run. This is good for importance sampling
    const size_t warmupCalls = static_cast<size_t>(0.2 * calls);
    gsl_monte_vegas_integrate(&gslFunction, integralLowerLimits, integralUpperLimits, dim, warmupCalls, gslWrapper::rng, gslState, &mean, &error);

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

        gsl_monte_vegas_integrate(&gslFunction, integralLowerLimits, integralUpperLimits, dim, calls, gslWrapper::rng, gslState, &mean, &error);

        currentTries++;
        if (currentTries >= maxTries)
        {
            // TODO return some error or warning flag
            break;
        }
    }

    gsl_monte_vegas_free(gslState);

    IntegrationResult result;
    result.result = mean;
    result.error = error;

    return result;
}

CollisionResultsGrid CollisionIntegral4::evaluateOnGrid(const IntegrationOptions& options, const CollisionTensorVerbosity& verbosity)
{
    CollisionMetadata metadata;
    metadata.basisSize = getPolynomialBasisSize();
    metadata.bStatisticalErrors = options.bIncludeStatisticalErrors;
    metadata.integrator = "Vegas Monte Carlo";
    metadata.basisName = "Chebyshev";
    metadata.seed = gSeedGSL;
    metadata.usedIntegrationOptions = options;

    CollisionResultsGrid result(mParticlePair, metadata);

    // ---- Setup progress reporting
    const bool bCanEverReportProgress = (verbosity.progressReportPercentage > 0 && verbosity.progressReportPercentage < 1.f);;
    uint32_t progressCounter = 0; // shared counter for all threads, resets every report interval
    const bool bNeedsTiming = verbosity.bPrintElapsedTime || bCanEverReportProgress;

    uint32_t reportInterval = 0;
    size_t totalIntegralCount = 0;
    size_t currentCount = 0; // updated only at checkpoints
    if (bCanEverReportProgress)
    {
        totalIntegralCount = countIndependentIntegrals();
        reportInterval = static_cast<uint32_t>(verbosity.progressReportPercentage * totalIntegralCount);
    }

    std::chrono::steady_clock::time_point startTime;
    if (bNeedsTiming)
    {
        startTime = std::chrono::steady_clock::now();
    }

    // Note symmetry: C[Tm(-rho_z), Tn(rho_par)] = (-1)^m C[Tm(rho_z), Tn(rho_par)]
    // which means we only need j <= N/2

    #pragma omp parallel
    {
        /* Each thread needs its collisionIntegral because evaluation of parsed matrix elements is not thread safe!
        Take thread-local copy inside the parallel region. Note that when using Python bindings this can lead to complications with the GIL:
        if anything in CollisionIntegral4 depends on Python state (eg. std::function callbacks obtained from Python),
        invoking copy operator means Python needs to update its reference count which in turn requires GIL to be held and prevents multithreading.
        Currently we avoid this by releasing GIL in Python-wrapped functions that call this function.
        See PythonBindings.cpp for more details. */
        CollisionIntegral4 workIntegral = *this;

        int threadID = 0;
        int numThreads = 1;
    #if WITH_OMP
        threadID = omp_get_thread_num();
        numThreads = omp_get_num_threads();
    #endif

        if (threadID == 0)
        {
            result.mMetadata.numThreads = numThreads;
        }

        // Limitation in older OMP implementations: for loops must use signed integer indices, size_t doesn't work
        const int32_t N = static_cast<int32_t>(getPolynomialBasisSize());

    #if WG_OMP_SUPPORTS_COLLAPSE
        #pragma omp for collapse(4)
    #else
        #pragma omp for
    #endif
        // m,n = Polynomial indices
        for (int32_t m = 2; m <= N; ++m)
        for (int32_t n = 1; n <= (N - 1); ++n)
        {
            // j,k = grid momentum indices 
            for (int32_t j = 1; j <= (N / 2); ++j)
            for (int32_t k = 1; k <= (N - 1); ++k)
            {

                const GridPoint gridPoint(
                    static_cast<uint32_t>(m),
                    static_cast<uint32_t>(n),
                    static_cast<uint32_t>(j),
                    static_cast<uint32_t>(k)
                );

                IntegrationResult localResult;

                // Integral vanishes if rho_z = 0 and m = odd. rho_z = 0 means j = N/2 which is possible only for even N
                if (2 * j == N && m % 2 != 0)
                {
                    localResult.result = 0.0;
                    localResult.error = 0.0;
                }
                else
                {
                    localResult = workIntegral.integrate(gridPoint, options);
                }

                result.updateValue(gridPoint, localResult.result, localResult.error);

                if (verbosity.bPrintEveryElement)
                {
                    // In principle this should be an OMP critical block, but probably not worth the slowdown
                    std::cout << "[Thread " << threadID << "]: " << "m=" << m << " n=" << n << " j=" << j << " k=" << k << " : "
                        << localResult.result << " +/- " << localResult.error << "\n";
                }

                if (bCanEverReportProgress)
                {
                    #pragma omp atomic
                    progressCounter++;
                }

                if (bCanEverReportProgress && threadID == 0)
                {
                    uint32_t progressCounter_atomic;
                    WG_PRAGMA_OMP_ATOMIC_READ
                    progressCounter_atomic = progressCounter;

                    if (progressCounter_atomic >= reportInterval)
                    {
                        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
                        auto elapsedTime = currentTime - startTime;
                        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(elapsedTime).count();

                        currentCount += progressCounter_atomic;
                        const double currentPercentage = static_cast<double>(currentCount) / static_cast<double>(totalIntegralCount);
                        // Estimate remaining duration by assuming linear progress
                        auto timeAtCompletion = (elapsedTime / currentPercentage);
                        auto timeRemaining = timeAtCompletion - elapsedTime;
                        auto secondsRemaining = std::chrono::duration_cast<std::chrono::seconds>(timeRemaining).count();
                       
                        std::cout << "Integral progress: " << currentCount << "/" << totalIntegralCount << " (" << 100.0 * currentPercentage << "%)." 
                            << " Time: " << elapsedSeconds << "s, remaining " << secondsRemaining << "s.\n";

                        // TODO replace on Windows, cannot be critical
                        WG_PRAGMA_OMP_ATOMIC_WRITE
                        progressCounter = 0;
                    }
                }

                // Check if we received instructions to stop
                if (threadID == 0 && utils::receivedExitSignal())
                {
                    std::exit(20);
                }

                } // end j,k
        } // end m,n

        // Fill in the j > N/2 elements
    #if WG_OMP_SUPPORTS_COLLAPSE
        #pragma omp for collapse(4)
    #else
        #pragma omp for
    #endif
        for (int32_t m = 2; m <= N; ++m)
        for (int32_t n = 1; n <= (N - 1); ++n)
        {
            for (int32_t j = N / 2 + 1; j <= (N - 1); ++j)
            for (int32_t k = 1; k <= (N - 1); ++k)
            {
                const GridPoint gridPoint(
                    static_cast<uint32_t>(m),
                    static_cast<uint32_t>(n),
                    static_cast<uint32_t>(j),
                    static_cast<uint32_t>(k)
                );

                const int32_t jOther = N - j;
                const int32_t sign = (m % 2 == 0 ? 1 : -1);

                const GridPoint otherPoint(
                    gridPoint.m,
                    gridPoint.n,
                    jOther,
                    gridPoint.k
                );

                const double val = sign * result.valueAt(otherPoint);
                const double err = options.bIncludeStatisticalErrors ? result.errorAt(otherPoint) : 0.0;

                result.updateValue(gridPoint, val, err);
            }
        }

    } // end #pragma omp parallel

    if (verbosity.bPrintElapsedTime)
    {
        std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
        auto elapsedTime = currentTime - startTime;
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(elapsedTime).count();
        std::cout << "(" << mParticlePair.first << ", " << mParticlePair.second << ") done, took " << elapsedSeconds << " seconds.\n\n";
    }

    return result;
}

void CollisionIntegral4::addCollisionElement(const CollisionElement<4> &elem)
{
    // TODO should check here that the collision element makes sense: has correct p1 particle etc
    if (elem.isUltrarelativistic())
    {
        collisionElements_ultrarelativistic.push_back(elem);
    }
    else
    {
        collisionElements_nonUltrarelativistic.push_back(elem);
    }
}

void CollisionIntegral4::handleModelChange(const ModelChangeContext& changeContext)
{
    for (auto& collisionElement : collisionElements_nonUltrarelativistic)
    {
        collisionElement.handleModelChange(changeContext);
    }
    for (auto& collisionElement : collisionElements_ultrarelativistic)
    {
        collisionElement.handleModelChange(changeContext);
    }
}

size_t CollisionIntegral4::countIndependentIntegrals() const
{
    const size_t N = getPolynomialBasisSize();
    // How many elements in each CollisionIntegral4
    size_t count = (N - 1) * (N - 1) * (N - 1) * (N - 1);

    // C[Tm(-x), Tn(y)] = (-1)^m C[Tm(x), Tn(y)]
    count = count / 2;
    // Take ceiling, avoiding type casts
    if (count % 2 != 0) count += 1;

    // Integral vanishes if rho_z = 0 and m = odd. rho_z = 0 means j = N/2 which is possible only for even N
    if (N % 2 == 0)
    {
        // how many odd m?
        size_t oddM= N / 2;
        count -= oddM;
    }
    return count;
}

bool CollisionIntegral4::isEmpty() const
{
    return collisionElements_nonUltrarelativistic.size() == 0
        && collisionElements_ultrarelativistic.size() == 0;
}

bool CollisionIntegral4::isValidGridPoint(const GridPoint& gridPoint) const
{
    return gridPoint.m >= 2 && gridPoint.m <= mBasisSize
        && gridPoint.n >= 1 && gridPoint.n < mBasisSize
        && gridPoint.j >= 1 && gridPoint.j < mBasisSize
        && gridPoint.k >= 1 && gridPoint.k < mBasisSize;
}

CollisionIntegral4::IntegrandParameters CollisionIntegral4::initializeIntegrandParameters(const GridPoint& gridPoint) const
{
    IntegrandParameters params;
    params.m = gridPoint.m;
    params.n = gridPoint.n;

    // Precalculate stuff related to the p1 momentum (optimization)
    params.rhoZ1 = chebyshev::rhoZGrid(gridPoint.j, mBasisSize);
    params.rhoPar1 = chebyshev::rhoParGrid(gridPoint.k, mBasisSize);
    params.pZ1 = chebyshev::rhoZ_to_pZ(params.rhoZ1);
    params.pPar1 = chebyshev::rhoPar_to_pPar(params.rhoPar1);
    params.TmTn_p1 = chebyshev::TmTn(gridPoint.m, gridPoint.n, params.rhoZ1, params.rhoPar1);
    params.p1 = std::sqrt(params.pZ1 * params.pZ1 + params.pPar1 * params.pPar1);
    return params;
}

std::vector<Kinematics> CollisionIntegral4::calculateKinematics(const CollisionElement<4> &CollisionElement, const InputsForKinematics& kinematicInput) const
{
    // Vacuum masses squared
    std::array<double, 4> massSquared;
    for (int i = 0; i < 4; ++i)
    {
        /* NB: we always take the cached mass squared here because the mass shouldn't depend on kinematics (ie. is momentum independent).
        Must revisit this if we ever let the mass be grid-dependent, including position dependence. */
        massSquared[i] =  CollisionElement.mExternalParticles[i].isUltrarelativistic() ? 0.0 : CollisionElement.mExternalParticles[i].getCachedMassSquared();
    }

    const double p1 = kinematicInput.p1;
    const double p2 = kinematicInput.p2;

    // Energies: Since p3 is not fixed yet we only know E1, E2
    const double E1 = std::sqrt(p1*p1 + massSquared[0]);
    const double E2 = std::sqrt(p2*p2 + massSquared[1]);

    // Now def. function g(p3) = FV4(p3) - msq4 and express the delta function in
    // terms of roots of g(p3) and delta(p3 - p3root); p3 integral becomes trivial.
    // Will need to solve a quadratic equation; some helper variables for it:
    const double Q = massSquared[0] + massSquared[1] + massSquared[2] - massSquared[3];
    const double kappa = Q + 2.0 * (E1 * E2 - kinematicInput.p1p2Dot);
    const double eps = 2.0 * (E1 + E2);
    const double delta = 2.0 * (kinematicInput.p1p3HatDot + kinematicInput.p2p3HatDot);

    // Quadratic eq. A p3^2 + B p3 + C = 0, take positive root(s)
    const double A = delta * delta - eps * eps;
    const double B = 2.0 * kappa * delta;
    const double C = kappa * kappa - eps * eps * massSquared[2];
    // Roots of g(p3):
    const double discriminant = B * B - 4.0 * A * C;

    if (discriminant < 0)
    {
        return std::vector<Kinematics>();
    }

    const double root1 = 0.5 * (-B - sqrt(discriminant)) / A;
    const double root2 = 0.5 * (-B + sqrt(discriminant)) / A;

#if WG_DEBUG
    // Check that g(p3) == 0 is reasonably satisfied 
    auto funcG = [&](double p3) {
        const double m3sq = massSquared[2];
        return kappa + delta*p3 - eps * sqrt(p3*p3 + m3sq);
    };

    assert(std::abs(funcG(root1)) < 1e-10 && std::abs(funcG(root2)) < 1e-10);
#endif

    // Since p3 is supposed to be magnitude, pick only positive roots (p3 = 0 contributes nothing to the integral)
    const std::vector<double> roots{root1, root2};

    std::vector<Kinematics> kinematics;
    kinematics.reserve(2);

    for (double p3 : roots) if (p3 > 0)
    {
        const double E3 = std::sqrt(p3 * p3 + massSquared[2]);

        // E4 is fixed by 4-momentum conservation. There is a theta(E4) so we only accept E4 >= 0
        const double E4 = E1 + E2 - E3;
        if (E4 < 0)
            continue;

        Kinematics newKinematics;

        // Make four vectors
        newKinematics.FV1 = FourVector(E1, kinematicInput.p1Vec);
        newKinematics.FV2 = FourVector(E2, kinematicInput.p2Vec);
        newKinematics.FV3 = FourVector(E3, p3 * kinematicInput.p3VecHat);

        newKinematics.FV4 = newKinematics.FV1 + newKinematics.FV2 - newKinematics.FV3;

        // g'(p3). Probably safe since we required p3 > 0, so E3 > 0:
        const double gDer = delta - eps * p3 / E3;

        newKinematics.prefactor = p2 * p3 * (p2 / (E2 + SMALL_NUMBER)) * (p3 / (E3 + SMALL_NUMBER)) / std::abs(gDer);

        kinematics.push_back(newKinematics);
    }

    return kinematics;
}

Kinematics CollisionIntegral4::calculateKinematics_ultrarelativistic(const InputsForKinematics& kinematicInput) const
{
    // Now E_i = p_i and the momentum-conserving delta function gives p3 = (p1*p2 - p1.p2 ) / (p1 + p2 - p3hat.(p1+p2))

    const double p1 = kinematicInput.p1;
    const double p2 = kinematicInput.p2;

    const double denom = p1 + p2 - kinematicInput.p1p3HatDot - kinematicInput.p2p3HatDot;
    const double p3 = (p1 * p2 - kinematicInput.p1p2Dot) / denom;

    // Probably should not happen?
    assert(p3 >= 0);

    Kinematics newKinematics;

    // Make four vectors
    newKinematics.FV1 = FourVector(p1, kinematicInput.p1Vec);
    newKinematics.FV2 = FourVector(p2, kinematicInput.p2Vec);
    newKinematics.FV3 = FourVector(p3, p3 * kinematicInput.p3VecHat);
    
    newKinematics.FV4 = newKinematics.FV1 + newKinematics.FV2 - newKinematics.FV3;

    // g'(p3)
    const double gDer = -2 * denom;

    newKinematics.prefactor = p2*p3 / std::abs(gDer);

    return newKinematics;
}


double CollisionIntegral4::evaluateCollisionElement(CollisionElement<4> &CollisionElement, const Kinematics &kinematics, const std::array<double, 4> &TmTn)
{
    // Need to set correct deltaF factors in CollisionElement
    std::array<double, 4> deltaF;
    for (int i = 0; i < 4; ++i)
    {
        if (CollisionElement.mExternalParticles[i].isInEquilibrium())
        {
            deltaF[i] = 0.0;
        }
        else
        {
            deltaF[i] = TmTn[i];
        }
    }

    // Evaluate |M|^2 P[f] for this CollisionElement
    const std::array<FourVector, 4> fourMomenta({kinematics.FV1, kinematics.FV2, kinematics.FV3, kinematics.FV4});
    double res = CollisionElement.evaluate(fourMomenta, deltaF);

    // Multiply by p2^2 / E2 * p3^2 / E3 |1/g'(p3)|
    res *= kinematics.prefactor;

    return res;
}

CollisionIntegral4::CollisionIntegral4(size_t polynomialBasisSize, const ParticleNamePair& particlePair)
    : mBasisSize(polynomialBasisSize),
    mParticlePair(particlePair)
{
}

void CollisionIntegral4::changePolynomialBasis(size_t newBasisSize)
{
    mBasisSize = newBasisSize;
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

    InputsForKinematics kinematicInput;
    kinematicInput.p1 = p1;
    kinematicInput.p2 = p2;

    // Create momentum 3-vectors
    const ThreeVector p1Vec(pPar1, 0, pZ1);
    const ThreeVector p2Vec(p2 * sinTheta2 * cosPhi2, p2 * sinTheta2 * sinPhi2, p2 * cosTheta2);

    // 'p3VecHat': like p3Vec, but normalized to 1. We will fix its magnitude using a delta(FV4^2 - msq4)
    const ThreeVector p3VecHat(sinTheta3 * cosPhi3, sinTheta3 * sinPhi3, cosTheta3);

    kinematicInput.p1Vec = p1Vec;
    kinematicInput.p2Vec = p2Vec;
    kinematicInput.p3VecHat = p3VecHat;

    // Precalculate some dot products of 3-vectors that are same for all CollElems
    kinematicInput.p1p2Dot = p1Vec * p2Vec;
    kinematicInput.p1p3HatDot = p1Vec * p3VecHat;
    kinematicInput.p2p3HatDot = p2Vec * p3VecHat;

    // Kinematics differs for each collision element since the masses are generally different.
    // But for CollElems with only ultrarelativistic (UR) particles the kinematic factors will always be the same

    bool bDoneUR = false;

    if (bOptimizeUltrarelativistic && collisionElements_ultrarelativistic.size() > 0)
    {
        Kinematics kinematics = calculateKinematics_ultrarelativistic(kinematicInput);

        // Account for theta(E4). Note that in the non-UR case this is built-in to the kinematics computation
        if (kinematics.FV4.energy() >= 0)
        {
            // Calculate polynomial factors (which our method uses as replacement for deltaF): Tm(rhoZ) Tn(rhoPar)
            // We're doing off-eq pair (a, l) so the rhoZ, rhoPar momenta here are always momenta of the l-particle.
            // This is automatically handled by bDeltaF flags in CollisionElement (although the logic is not very transparent)
            const std::array<double, 4> TmTn {
                integrandParameters.TmTn_p1,
                chebyshev::TmTn(m, n, kinematics.FV2),
                chebyshev::TmTn(m, n, kinematics.FV3),
                chebyshev::TmTn(m, n, kinematics.FV4)
            };
            
            for (CollisionElement<4> &CollisionElement : collisionElements_ultrarelativistic)
            {
                fullIntegrand += evaluateCollisionElement(CollisionElement, kinematics, TmTn);
            }
        }
        bDoneUR = true;
    }

    //---- Non-ultrarelativistic computations. Here each CollisionElement has its own kinematics and the momentum delta-function needs a sum over p3 roots

    // ugly lambda to avoid copy-paste or unnecessary member function just for this
    auto evaluateNonUR = [&](CollisionElement<4> &CollisionElement)
    {
        double res = 0.0;

        const std::vector<Kinematics> kinematicFactors = calculateKinematics(CollisionElement, kinematicInput);

        // kinematicFactors only contains solutions with E4 > 0 so the theta(E4) is OK.
        for (const Kinematics &kinematics : kinematicFactors)
        {
            // Calculate polynomial factors (which our method uses as replacement for deltaF): Tm(rhoZ) Tn(rhoPar)
            const std::array<double, 4> TmTn {
                integrandParameters.TmTn_p1,
                chebyshev::TmTn(m, n, kinematics.FV2),
                chebyshev::TmTn(m, n, kinematics.FV3),
                chebyshev::TmTn(m, n, kinematics.FV4)
            };

            res += evaluateCollisionElement(CollisionElement, kinematics, TmTn);
        }
        return res;
    };

    for (CollisionElement<4> &CollisionElement : collisionElements_nonUltrarelativistic) fullIntegrand += evaluateNonUR(CollisionElement);
    
    if (!bDoneUR) for (CollisionElement<4> &CollisionElement : collisionElements_ultrarelativistic) fullIntegrand += evaluateNonUR(CollisionElement);

    // Common numerical prefactor
    constexpr double PI = constants::pi;
    constexpr double pi2Pow58 = (2.0 * PI) * (2.0 * PI) * (2.0 * PI) * (2.0 * PI) * (2.0 * PI) * 8.0;
    return fullIntegrand / pi2Pow58;
}

} // namespace