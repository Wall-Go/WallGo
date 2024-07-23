#include "CollisionTensor.h"

#include <array>
#include <fstream>
#include <sstream>
#include <regex> // Reading matrix elements from file
#include <algorithm> // std::remove_if
#include <chrono>
#include <filesystem>

namespace wallgo
{


CollisionTensor::CollisionTensor()
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
}

CollisionTensor::CollisionTensor(size_t basisSize)
    : CollisionTensor()
{
    changePolynomialBasisSize(basisSize);
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


CollisionIntegral4 CollisionTensor::setupCollisionIntegral(const std::shared_ptr<ParticleSpecies>& particle1, const std::shared_ptr<ParticleSpecies>& particle2, 
    const std::string &matrixElementFile, size_t inBasisSize, bool bVerbose)
{
    std::vector<CollisionElement<4>> collisionElements = parseMatrixElements(particle1->getName(), particle2->getName(), matrixElementFile, bVerbose);

    ParticleNamePair namePair(particle1->name, particle2->name);
    CollisionIntegral4 collisionIntegral(inBasisSize, namePair);

    for (const CollisionElement<4> &elem : collisionElements)
    {
        collisionIntegral.addCollisionElement(elem);
    }

    return collisionIntegral;
}


void CollisionTensor::setupCollisionIntegrals(const std::filesystem::path& matrixElementFile, bool bVerbose)
{
    clearIntegralCache();

    for (const std::shared_ptr<ParticleSpecies>& particle1 : outOfEqParticles)
    for (const std::shared_ptr<ParticleSpecies>& particle2 : outOfEqParticles)
    {
        const auto namePair = std::make_pair(particle1->getName(), particle2->getName());

        CollisionIntegral4 newIntegral = setupCollisionIntegral(particle1, particle2, matrixElementFile.string(), mBasisSize, bVerbose);

        mCachedIntegrals.insert({ namePair, newIntegral });
    }
}

void CollisionTensor::clearIntegralCache()
{
    mCachedIntegrals.clear();
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

        //std::cout << particlePairToString(namePair) << " done\n";

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


size_t CollisionTensor::countIndependentIntegrals() const
{
    size_t res = 0;
    for (const auto& [_, integral] : mCachedIntegrals)
    {
        res += integral.countIndependentIntegrals();
    }
    return res;
}

/*
CollisionElement<4> CollisionTensor::makeCollisionElement(const std::string &particleName2, const std::vector<size_t> &indices, 
    const std::string &expr, const std::vector<std::string> &symbols)
{
    assert(indices.size() == 4);

    const std::shared_ptr<ParticleSpecies> p1 = particles[indices[0]];
    const std::shared_ptr<ParticleSpecies> p2 = particles[indices[1]];
    const std::shared_ptr<ParticleSpecies> p3 = particles[indices[2]];
    const std::shared_ptr<ParticleSpecies> p4 = particles[indices[3]];
    
    CollisionElement<4> collisionElement( { p1, p2, p3, p4} );

    std::map<std::string, double> variables;
    for (const std::string& s : symbols)
    {
        if (!mModelParameters.contains(s))
        {
            std::cerr << "CollisionTensor error: symbol " << s << " not found in modelParameters map, can't parse matrix element " << expr << "\n";
            return collisionElement;
        }
        variables[s] = mModelParameters.getParameterValue(s);
    }

    collisionElement.matrixElement.initSymbols(variables);
    // Parse expr as a math function
    collisionElement.matrixElement.setExpression(expr);

    // Set deltaF flags: in general there can be 4 deltaF terms but we only take the ones with deltaF of particle2
    for (size_t i = 0; i < collisionElement.bDeltaF.size(); ++i) 
    {
        collisionElement.bDeltaF[i] = (collisionElement.particles[i]->getName() == particleName2);
    }
    
    return collisionElement;
}
*/


} // namespace