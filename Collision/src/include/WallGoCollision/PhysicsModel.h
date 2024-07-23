#pragma once

#include <string>
#include <unordered_map>
#include <type_traits>
#include <cassert>
#include <memory>
#include <map>

#include "Common.h"
#include "MatrixElement.h"
#include "ModelParameters.h"
#include "ParticleSpecies.h"
#include "CollisionTensor.h"

namespace wallgo
{

class PhysicsModel
{
public:

    /* Defines a new particle species for the model.
    * Usage:
    *   - description is a ParticleDescription struct that must be filled in accordingly
    *   - description.name must be a unique string name
    *   - index must be a unique integer identifier for the particle and must match the intended index in matrix element files.
    * 
    * If the input is invalid, return value is false and no particle is defined.
    */
    bool defineParticleSpecies(const ParticleDescription& description, uint32_t index);

    /* Read matrix elements from a file and stores them internally.
    This will only consider expressions where at least one currently registered out-of-equilibrium particle appears as an external particle.
    Note that this function clears any previously stored matrix elements.
    Returns false if something goes wrong. */
    bool readMatrixElements(
        const std::filesystem::path& matrixElementFile,
        bool bPrintMatrixElements = false);

    // Get copy of our cached matrix elements, grouped by out-of-equilibrium particle indices
    std::map<IndexPair, std::vector<MatrixElement>> getMatrixElements() { return mMatrixElements; }

    
    CollisionTensor makeCollisionTensor(size_t basisSize);


    // Models cannot be copied, only moving ownership is allowed
    PhysicsModel(const PhysicsModel&) = delete;
    PhysicsModel operator=(const PhysicsModel&) = delete;

    PhysicsModel(PhysicsModel&&) noexcept = default;
    PhysicsModel& operator=(PhysicsModel&&) noexcept = default;

private:

    ModelParameters mParameters;
    
    /* Use custom indices for particle lookup. unique_ptr because the model owns particles,
    and we will pass particle references around to collision elements as raw pointers. */
    std::unordered_map<uint32_t, std::unique_ptr<ParticleSpecies>> mParticles;

    ParticleNameMap mParticleNameMap;
    // Indices of out-of-equilibrium particles
    std::vector<uint32_t> offEqIndices;

    // Cached matrix elements. This is std::map instead of std::unordered_map because we haven't defined a hash for IndexPair
    std::map<IndexPair, std::vector<MatrixElement>> mMatrixElements;

    // Must be called after changing mParameters to propagate changes to cached particle masses
    void updateParticleMassCache();

    void printMatrixElements() const;

};



} // namespace
