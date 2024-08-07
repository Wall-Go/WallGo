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

bool ModelDefinition::defineParticleSpecies(const ParticleDescription& description)
{
    // Check if particle with this index or name has already been defined
    for (const ParticleDescription& knownParticle : mParticleDescriptions)
    {
        if (knownParticle.name == description.name)
        {
            std::cerr << "Particle definition error: particle named " << description.name << " has already been defined and has index: " << knownParticle.index << std::endl;
            return false;
        }

        if (knownParticle.index == description.index)
        {
            std::cerr << "Particle definition error: particle at index " << description.index << " has already been defined and is named: " << knownParticle.index << std::endl;
            return false;
        }
    }

    mParticleDescriptions.push_back(description);
    return true;
}

void ModelDefinition::defineParameter(const char* symbol, double value)
{
    defineParameter(std::string(symbol), value);
}

void ModelDefinition::defineParameter(const std::string& symbol, double value)
{
    if (mParameters.contains(symbol))
    {
        std::cerr << "Attempted to redefine already defined parameter: " << symbol << "\n";
        return;
    }

    // These are reserved:
    if (symbol == "s" || symbol == "t" || symbol == "u")
    {
        std::cerr << "Parameter name " << symbol << " is reserved for internal use, please choose a different symbol\n";
        return;
    }

    mParameters.addOrModifyParameter(symbol, value);
}

void ModelDefinition::defineParameters(const ModelParameters& inParams)
{
    for (const auto& [symbol, value] : inParams.getParameterMap())
    {
        defineParameter(symbol, value);
    }
}


PhysicsModel::PhysicsModel(const ModelDefinition& modelDefinition)
{
    // This assumes that particle and parameter contents has been validated already (currently done by ModelDefinition itself)

    mParameters = modelDefinition.mParameters;
    for (const ParticleDescription& particle : modelDefinition.mParticleDescriptions)
    {
        mParticles.emplace(particle.index, ParticleSpecies(particle));
        mParticleNameMap.insert({ particle.name, particle.index });

        if (!particle.bInEquilibrium)
        {
            mOffEqIndices.push_back(particle.index);
        }
    }
}

PhysicsModel::~PhysicsModel()
{
    for (IModelObserver* observer : mObservers)
    {
        observer->handleModelDestruction();
    }
}

PhysicsModel::PhysicsModel(const PhysicsModel& other)
{
    // Copy everything but the observer registrations
    mObservers.clear();
    
    mParameters = other.mParameters;
    mParticles = other.mParticles;
    mParticleNameMap = other.mParticleNameMap;
    mOffEqIndices = other.mOffEqIndices;
    mMatrixElements = other.mMatrixElements;
}
PhysicsModel& PhysicsModel::operator=(const PhysicsModel& other)
{
    if (&other != this)
    {
        // Copy everything but the observer registrations

        mParameters = other.mParameters;
        mParticles = other.mParticles;
        mParticleNameMap = other.mParticleNameMap;
        mOffEqIndices = other.mOffEqIndices;
        mMatrixElements = other.mMatrixElements;
    }
    return *this;
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
    
    ModelChangeContext changeContext;
    changeContext.changedParams.addOrModifyParameter(symbol, value);
    changeContext.changedParticles = computeParticleChanges();

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

    ModelChangeContext changeContext;
    changeContext.changedParams = newValues;
    changeContext.changedParticles = computeParticleChanges();

    notifyModelChange(changeContext);
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

void PhysicsModel::registerObserver(IModelObserver& observer)
{
    mObservers.push_back(&observer);
}

void PhysicsModel::unregisterObserver(IModelObserver& observer)
{
    auto it = std::find(mObservers.begin(), mObservers.end(), &observer);
    if (it != mObservers.end())
    {
        mObservers.erase(it);
    }
}

void PhysicsModel::printMatrixElements() const
{
    for (const auto& [indexPair, elements] : mMatrixElements)
    {
        std::string_view name1 = mParticles.at(indexPair.first).getName();
        std::string_view name2 = mParticles.at(indexPair.second).getName();

        std::cout << "\n## (" << name1 << ", " << name2 << ") matrix elements ##\n\n";

        for (const MatrixElement& m : elements)
        {
            std::vector<uint32_t> indices = m.getParticleIndices();
            std::cout << "[";
            for (uint32_t i = 0; i < indices.size(); ++i)
            {
                std::cout << mParticles.at(indices[i]).getName();
                if (i != indices.size() - 1) std::cout << ", ";
            }

            std::cout << "] : " << m.getExpression() << "\n";
        }
    }
}

std::vector<ParticleChangeContext> PhysicsModel::computeParticleChanges()
{
    std::vector<ParticleChangeContext> outContext;
    outContext.reserve(mParticles.size());
    for (auto& [index, particle] : mParticles)
    {
        ParticleChangeContext context;
        context.particleIndex = index;
        // Re-evaluate particle mass and let the particle cache it
        context.newMassSq = particle.computeMassSquared(mParameters, false, true);
        outContext.push_back(context);
    }
    return outContext;
}

void PhysicsModel::notifyModelChange(const ModelChangeContext& context) const
{
    for (IModelObserver* observer : mObservers)
    {
        observer->handleModelChange(context);
    }
}

CollisionTensor PhysicsModel::createCollisionTensor(size_t basisSize, const std::vector<uint32_t>& offEqParticleIndices)
{
    // Sanity checks
    for (uint32_t idx : offEqParticleIndices)
    {
        if (mParticles.count(idx) == 0)
        {
            std::cerr << "Unknown particle index: " << idx << std::endl;
            return CollisionTensor(this);
        }

        if (mParticles.at(idx).isInEquilibrium())
        {
            std::cerr << "Attempted to create collision tensor for particle " << mParticles.at(idx).getName()
                << " (index " << idx << "), but the particle is in equilibrium" << std::endl;
            return CollisionTensor(this);
        }
    }

    CollisionTensor outTensor(this, basisSize);


    for (uint32_t idx1 : offEqParticleIndices) for (uint32_t idx2 : offEqParticleIndices)
    {
        IndexPair indexPair(idx1, idx2);
        ParticleNamePair namePair(mParticles.at(idx1).getName(), mParticles.at(idx2).getName());

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

    auto name1 = mParticles.at(offEqIndices.first).getName();
    auto name2 = mParticles.at(offEqIndices.second).getName();

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

    // Copy ParticleSpecies objects to the CollisionElement
    std::array<ParticleSpecies, 4> externalParticles;

    for (uint32_t i = 0; i < indices.size(); ++i)
    {
        const uint32_t particleIndex = indices[i];
        externalParticles[i] = mParticles.at(particleIndex);
    }

    return CollisionElement<4>(externalParticles, matrixElement, bDeltaF);
}


} // namespace wallgo
