#include "ModelObserver.h"
#include "PhysicsModel.h"
#include "ParticleSpecies.h"
#include "CollisionElement.h"
#include "MatrixElementParsing.h"

#include <algorithm>
#include <array>
#include <iostream>

namespace wallgo
{

bool PhysicsModel::defineParticleSpecies(const ParticleDescription& description)
{
    if (isLocked())
    {
        std::cerr << "Attempted to define a new particle to a locked wallgo::PhysicsModel!\n";
        return false;
    }

    // Check if particle with this index has already been defined
    if (mParticles.count(description.index) > 0)
    {
        assert(mParticles.at(description.index));
        std::string_view otherName = mParticles.at(description.index)->getName();
        std::cerr << "Particle definition error: particle at index " << description.index << " has already been defined and is named: " << otherName << std::endl;
        return false;
    }

    // Check if particle with this name has already been defined
    if (mParticleNameMap.count(description.name) > 0)
    {
        const uint32_t otherIndex = mParticleNameMap.at(description.name);
        std::cerr << "Particle definition error: particle named " << description.name << " has already been defined and has index: " << otherIndex << std::endl;
        return false;
    }

    mParticles.emplace(description.index, std::make_unique<ParticleSpecies>(description));

    mParticleNameMap.insert({ description.name, description.index });

    if (!description.bInEquilibrium)
    {
        mOffEqIndices.push_back(description.index);
    }

    return true;
}

void PhysicsModel::defineParameter(const char* symbol, double value)
{
    defineParameter(std::string(symbol), value);
}

void PhysicsModel::defineParameter(const std::string& symbol, double value)
{
    if (isLocked())
    {
        std::cerr << "Attempted to define a new model parameter to a locked wallgo::PhysicsModel!!\n";
        return;
    }

    if (mParameters.contains(symbol))
    {
        std::cerr << "Attempted to redefine already defined parameter: " << symbol << "\n";
        return;
    }
    mParameters.addOrModifyParameter(symbol, value);
}

void PhysicsModel::defineParameters(const ModelParameters& inParams)
{
    for (const auto& [symbol, value] : inParams.getParameterMap())
    {
        defineParameter(symbol, value);
    }
}

void PhysicsModel::updateParameter(const char* symbol, double value)
{
    updateParameter(std::string(symbol), value);
}

void PhysicsModel::updateParameter(const std::string& symbol, double value)
{
    if (!mParameters.contains(symbol))
    {
        std::cerr << "Attempted to update undefined parameter: " << symbol << "\n";
        return;
    }
    mParameters.addOrModifyParameter(symbol, value);

    updateParticleMassCache();
    
    ModelChangeContext changeContext;
    changeContext.changedParams.addOrModifyParameter(symbol, value);
    notifyModelChange(changeContext);
}

void PhysicsModel::updateParameters(const ModelParameters& newValues)
{
    for (const auto& [symbol, value] : newValues.getParameterMap())
    {
        if (!mParameters.contains(symbol))
        {
            std::cerr << "Attempted to update undefined parameter: " << symbol << "\n";
        }
        mParameters.addOrModifyParameter(symbol, value);
    }

    updateParticleMassCache();

    ModelChangeContext changeContext;
    changeContext.changedParams = newValues;
    notifyModelChange(changeContext);
}

void PhysicsModel::lockModelDefinitions()
{
    bLocked = true;

    // Can now cache model-specific stuff
    updateParticleMassCache();
}

bool PhysicsModel::readMatrixElements(
    const std::filesystem::path& matrixElementFile,
    bool bPrintMatrixElements)
{
    mMatrixElements.clear();

    const bool bReadOK = utils::parseMatrixElements(matrixElementFile, mOffEqIndices, mParameters.getParameterMap(), mMatrixElements);
    if (!bReadOK) return false;

    if (bPrintMatrixElements)
    {
        printMatrixElements();
    }

    return true;
}

CollisionTensor PhysicsModel::createCollisionTensor(size_t basisSize)
{
    return createCollisionTensor(basisSize, mOffEqIndices);
}


void PhysicsModel::updateParticleMassCache()
{
    for (auto& [_, particle] : mParticles)
    {
        assert(particle);
        particle->computeMassSquared(mParameters, false, true);
    }
}

void PhysicsModel::printMatrixElements() const
{
    for (const auto& [indexPair, elements] : mMatrixElements)
    {
        std::string_view name1 = mParticles.at(indexPair.first)->getName();
        std::string_view name2 = mParticles.at(indexPair.second)->getName();

        std::cout << "\n## (" << name1 << ", " << name2 << ") matrix elements ##\n\n";

        for (const MatrixElement& m : elements)
        {
            std::vector<uint32_t> indices = m.getParticleIndices();
            std::cout << "[";
            for (uint32_t i = 0; i < indices.size(); ++i)
            {
                std::cout << mParticles.at(indices[i])->getName();
                if (i != indices.size() - 1) std::cout << ", ";
            }

            std::cout << "] : " << m.getExpression() << "\n";
        }
    }
}

void PhysicsModel::registerObserver(const IModelObserver* observer)
{
    if (observer && std::find(mObservers.begin(), mObservers.end(), observer) != mObservers.end())
    {
        mObservers.emplace_back(observer);
    }
    else
    {
        std::cerr << "Warning: redundant registerObserver() call\n";
    }
}

void PhysicsModel::unregisterObserver(const IModelObserver* observer)
{
    if (!observer) return;

    auto it = std::find(mObservers.begin(), mObservers.end(), observer);
    if (it != mObservers.end())
    {
        mObservers.erase(it);
    }
}

void PhysicsModel::notifyModelChange(const ModelChangeContext& context) const
{
    for (IModelObserver* observer : mObservers)
    {
        observer->handleModelChange(context);
    }
}

CollisionTensor PhysicsModel::createCollisionTensor(size_t basisSize, const std::vector<uint32_t>& offEqParticleIndices) const
{
    // Sanity checks
    for (uint32_t idx : offEqParticleIndices)
    {
        if (mParticles.count(idx) == 0)
        {
            std::cerr << "Unknown particle index: " << idx << std::endl;
            return CollisionTensor(this);
        }

        if (mParticles.at(idx)->isInEquilibrium())
        {
            std::cerr << "Attempted to create collision tensor for particle " << mParticles.at(idx)->getName()
                << " (index " << idx << "), but the particle is in equilibrium" << std::endl;
            return CollisionTensor(this);
        }
    }

    CollisionTensor outTensor(this, basisSize);


    for (uint32_t idx1 : offEqParticleIndices) for (uint32_t idx2 : offEqParticleIndices)
    {
        IndexPair indexPair(idx1, idx2);
        ParticleNamePair namePair(mParticles.at(idx1)->getName(), mParticles.at(idx2)->getName());

        outTensor.addCollisionIntegral(namePair, createCollisionIntegral4(basisSize, indexPair));
    }

    return outTensor;
}

CollisionIntegral4 PhysicsModel::createCollisionIntegral4(size_t basisSize, const IndexPair& offEqIndices) const
{
#if WG_DEBUG
    const bool bFound1 = std::find(mOffEqIndices.begin(), mOffEqIndices.end(), offEqIndices.first) != mOffEqIndices.end();
    const bool bFound2 = std::find(mOffEqIndices.begin(), mOffEqIndices.end(), offEqIndices.second) != mOffEqIndices.end();

    assert(bFound1 && "Off eq particle 1 not found");
    assert(bFound2 && "Off eq particle 2 not found");
#endif

    auto name1 = mParticles.at(offEqIndices.first)->getName();
    auto name2 = mParticles.at(offEqIndices.second)->getName();

    CollisionIntegral4 outIntegral(basisSize, ParticleNamePair(name1, name2));

    // Fill in the collision integral with CollisionElements

    assert(mMatrixElements.count(offEqIndices) > 0);
    
    for (const MatrixElement& matrixElement : mMatrixElements.at(offEqIndices))
    {
        outIntegral.addCollisionElement(createCollisionElement(offEqIndices, matrixElement));
    }

    return outIntegral;
}

CollisionElement<4> PhysicsModel::createCollisionElement(const IndexPair& offEqIndices, const MatrixElement& matrixElement) const
{
    /* To create correct CollisionElement we need to specify which delta_f terms it should include in the statistical part.
    * Our matrix elements are of form M[a,b,c,d] where the indices specify "positions" of the external particles,
    * and the assumed setup is that a == offEqIndices.first. We only include delta_f terms for the second off-eq particle,
    * so need to find which "position" indices are this particle:
    */

    std::vector<uint32_t> indices = matrixElement.getParticleIndices();
    assert(indices.size() == 4);
    // Cannot happen if matrix elements are setup properly:
    assert(indices[0] == offEqIndices.first && "Invalid particle1 index in matrix element");

    std::array<bool, 4> bDeltaF;

    bool bFoundAny = false;

    for (uint32_t i = 0; i < bDeltaF.size(); ++i)
    {
        bDeltaF[i] = (indices[i] == offEqIndices.second);
        bFoundAny |= bDeltaF[i];
    }
    assert(bFoundAny && "Matrix element had no particle2");

    std::array<const ParticleSpecies*, 4> externalParticles{ nullptr, nullptr, nullptr, nullptr };

    for (uint32_t i = 0; i < indices.size(); ++i)
    {
        const uint32_t particleIndex = indices[i];
        externalParticles[i] = mParticles.at(particleIndex).get();
    }

    return CollisionElement<4>(externalParticles, matrixElement, bDeltaF);
}

} // namespace wallgo
