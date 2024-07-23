#include "PhysicsModel.h"
#include "ParticleSpecies.h"
#include "CollisionElement.h"
#include "MatrixElementParsing.h"

#include <iostream>

namespace wallgo
{

bool PhysicsModel::defineParticleSpecies(const ParticleDescription& description, uint32_t index)
{
    // Check if particle with this index has already been defined
    if (mParticles.count(index) > 0)
    {
        assert(mParticles.at(index));
        std::string_view otherName = mParticles.at(index)->getName();
        std::cerr << "Particle definition error: particle at index " << index << " has already been defined and is named: " << otherName << std::endl;
        return false;
    }

    // Check if particle with this name has already been defined
    if (mParticleNameMap.count(description.name) > 0)
    {
        const uint32_t otherIndex = mParticleNameMap.at(description.name);
        std::cerr << "Particle definition error: particle named " << description.name << " has already been defined and has index: " << otherIndex << std::endl;
        return false;
    }

    mParticles.emplace(index, std::make_unique<ParticleSpecies>(description));

    mParticleNameMap.insert({ description.name, index });

    if (!description.bInEquilibrium)
    {
        offEqIndices.push_back(index);
    }

    return true;
}

bool PhysicsModel::readMatrixElements(
    const std::filesystem::path& matrixElementFile,
    bool bPrintMatrixElements)
{
    mMatrixElements.clear();

    const bool bReadOK = utils::parseMatrixElements(matrixElementFile, offEqIndices, mParameters.getParameterMap(), mMatrixElements);
    if (!bReadOK) return false;

    if (bPrintMatrixElements)
    {
        printMatrixElements();
    }

    return true;
}

CollisionTensor PhysicsModel::makeCollisionTensor(size_t basisSize)
{
    CollisionTensor outTensor(basisSize);

    outTensor.
    

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

        std::cout << "## (" << name1 << ", " << name2 << ") matrix elements ##\n";

        for (const MatrixElement& m : elements)
        {
            std::vector<uint32_t> indices = m.getParticleIndices();
            std::cout << "[";
            for (uint32_t i : indices)
            {
                std::cout << mParticles.at(i)->getName() << ", ";
            }

            std::cout << "] : " << m.getExpression() << "\n";
        }
    }
}

}
