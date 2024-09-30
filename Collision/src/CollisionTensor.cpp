#include "CollisionTensor.h"
#include "PhysicsModel.h"

#include <array>
#include <fstream>
#include <sstream>
#include <regex> // Reading matrix elements from file
#include <algorithm> // std::remove_if
#include <chrono>
#include <filesystem>

namespace wallgo
{


CollisionTensor::CollisionTensor(PhysicsModel* creator)
{
    // Set default options

    mBasisSize = 1;

    mDefaultIntegrationOptions.calls = 50000;
    mDefaultIntegrationOptions.maxIntegrationMomentum = 20;
    mDefaultIntegrationOptions.maxTries = 50;
    mDefaultIntegrationOptions.relativeErrorGoal = 1e-1;
    mDefaultIntegrationOptions.absoluteErrorGoal = 1e-8;
    mDefaultIntegrationOptions.bOptimizeUltrarelativistic = true;
    mDefaultIntegrationOptions.bIncludeStatisticalErrors = true;

    mDefaultVerbosity = CollisionTensorVerbosity();

    assert(creator);
    mObservingModel = creator;
    creator->registerObserver(*this);
}

CollisionTensor::CollisionTensor(PhysicsModel* creator, size_t basisSize)
    : CollisionTensor(creator)
{
    changePolynomialBasisSize(basisSize);
}

CollisionTensor::~CollisionTensor()
{
    if (mObservingModel)
    {
        mObservingModel->unregisterObserver(*this);
    }
}

CollisionTensor::CollisionTensor(const CollisionTensor& other)
{
    mDefaultIntegrationOptions = other.mDefaultIntegrationOptions;
    mDefaultVerbosity = other.mDefaultVerbosity;

    mCachedIntegrals = other.mCachedIntegrals;
    changePolynomialBasisSize(other.mBasisSize);

    mObservingModel = other.mObservingModel;
    if (mObservingModel)
    {
        mObservingModel->registerObserver(*this);
    }
}

CollisionTensor& CollisionTensor::operator=(const CollisionTensor& other)
{
    if (&other == this) return *this;
    
    mDefaultIntegrationOptions = other.mDefaultIntegrationOptions;
    mDefaultVerbosity = other.mDefaultVerbosity;

    mCachedIntegrals = other.mCachedIntegrals;
    changePolynomialBasisSize(other.mBasisSize);

    mObservingModel = other.mObservingModel;
    if (mObservingModel)
    {
        mObservingModel->registerObserver(*this);
    }

    return *this;
}

void CollisionTensor::setDefaultIntegrationOptions(const IntegrationOptions& options)
{
    mDefaultIntegrationOptions = options;
}

void CollisionTensor::setDefaultIntegrationVerbosity(const CollisionTensorVerbosity& verbosity)
{
    mDefaultVerbosity = verbosity;
}


void CollisionTensor::changePolynomialBasisSize(size_t newBasisSize)
{
    mBasisSize = newBasisSize;
    for (auto& [key, integral] : mCachedIntegrals)
    {
        integral.changePolynomialBasis(newBasisSize);
    }
}

void CollisionTensor::addCollisionIntegral(const ParticleNamePair& particleNames, const CollisionIntegral4& inIntegral)
{
    mCachedIntegrals[particleNames] = inIntegral;
}

CollisionIntegral4* CollisionTensor::getIntegralForPair(const ParticleNamePair& particleNames)
{
    if (mCachedIntegrals.count(particleNames) > 0)
    {
        return &mCachedIntegrals.at(particleNames);
    }
    else
    {
        return nullptr;
    }
}


CollisionResultsGrid CollisionTensor::computeIntegralsForPair(
    const std::string& particle1,
    const std::string& particle2,
    const IntegrationOptions& options,
    const CollisionTensorVerbosity& verbosity)
{
    const auto pairName = std::make_pair(particle1, particle2);
    if (mCachedIntegrals.count(pairName) < 1)
    {
        std::cerr << "No cached collision integral found for particle pair: ("
            << particle1 << ", " << particle2 << ")" << std::endl;
        
        assert(false && "Particle pair not found");
        
        return CollisionResultsGrid(ParticleNamePair("Unknown", "Unknown"), CollisionMetadata());
    }

    return mCachedIntegrals.at(pairName).evaluateOnGrid(options, verbosity);
}

CollisionResultsGrid CollisionTensor::computeIntegralsForPair(
    const std::string& particle1,
    const std::string& particle2,
    const CollisionTensorVerbosity& verbosity)
{
    return computeIntegralsForPair(particle1, particle2, mDefaultIntegrationOptions, verbosity);
}

CollisionResultsGrid CollisionTensor::computeIntegralsForPair(
    const std::string& particle1,
    const std::string& particle2,
    const IntegrationOptions& options)
{
    return computeIntegralsForPair(particle1, particle2, options, mDefaultVerbosity);
}

CollisionResultsGrid CollisionTensor::computeIntegralsForPair(
    const std::string& particle1,
    const std::string& particle2)
{
    return computeIntegralsForPair(particle1, particle2, mDefaultIntegrationOptions, mDefaultVerbosity);
}

CollisionTensorResult CollisionTensor::computeIntegralsAll(const IntegrationOptions& options, const CollisionTensorVerbosity& verbosity)
{
    if (mCachedIntegrals.size() < 1)
    {
        std::cerr << "Warning: computeIntegralsAll() called on empty CollisionTensor object" << std::endl;
        // return empty result
        return CollisionTensorResult();
    }

    CollisionTensorResult result(mCachedIntegrals.size());

    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    if (verbosity.bPrintElapsedTime)
    {
        startTime = std::chrono::steady_clock::now();
    }

    size_t i = 0;
    for (auto& [namePair, integral] : mCachedIntegrals)
    {
        result.mData[i] = integral.evaluateOnGrid(options, verbosity);
        ++i;
    }

    if (verbosity.bPrintElapsedTime)
    {
        auto endTime = std::chrono::steady_clock::now();
        auto elapsedTime = endTime - startTime;
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(elapsedTime).count();

        std::cout << "\nAll done, took " << elapsedSeconds << " seconds.\n" << std::endl;
    }

    return result;
}

CollisionTensorResult CollisionTensor::computeIntegralsAll(const IntegrationOptions& options)
{
    return computeIntegralsAll(options, mDefaultVerbosity);
}

CollisionTensorResult CollisionTensor::computeIntegralsAll(const CollisionTensorVerbosity& verbosity)
{
    return computeIntegralsAll(mDefaultIntegrationOptions, verbosity);
}

CollisionTensorResult CollisionTensor::computeIntegralsAll()
{
    return computeIntegralsAll(mDefaultIntegrationOptions, mDefaultVerbosity);
}

IntegrationResult CollisionTensor::computeSingleIntegral(const std::string& particle1, const std::string& particle2, const GridPoint& gridPoint, const IntegrationOptions& options)
{
    const auto pairName = std::make_pair(particle1, particle2);
    if (mCachedIntegrals.count(pairName) < 1)
    {
        std::cerr << "No cached collision integral found for particle pair: ("
            << particle1 << ", " << particle2 << ")" << std::endl;

        assert(false && "Particle pair not found");

        IntegrationResult res;
        res.result = 0.0;
        res.error = 0.0;
        return res;
    }

    return mCachedIntegrals.at(pairName).integrate(gridPoint, options);
}

IntegrationResult CollisionTensor::computeSingleIntegral(const std::string& particle1, const std::string& particle2, const GridPoint& gridPoint)
{
    return computeSingleIntegral(particle1, particle2, gridPoint, mDefaultIntegrationOptions);
}


size_t CollisionTensor::countIndependentIntegrals() const
{
    size_t res = 0;
    for (const auto& [_, integral] : mCachedIntegrals)
    {
        res += integral.countIndependentIntegrals();
    }
    return res;
}

void CollisionTensor::handleModelChange(const ModelChangeContext& context)
{
    for (auto& [_, integral] : mCachedIntegrals)
    {
        integral.handleModelChange(context);
    }
}

void CollisionTensor::handleModelDestruction()
{
    mObservingModel = nullptr;
}

} // namespace
