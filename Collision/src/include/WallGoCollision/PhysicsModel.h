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

struct ModelChangeContext;
class IModelObserver;

class PhysicsModel
{
public:

    /* Defines a new particle species for the model.
    * Usage:
    *   - description is a ParticleDescription struct that must be filled in accordingly
    *   - description.name must be a unique string name
    *   - description.index must be a unique integer identifier for the particle and must match the intended index in matrix element files .
    * 
    * If the input is invalid, return value is false and no particle is defined.
    */
    bool defineParticleSpecies(const ParticleDescription& description);

    // Defines a new symbolic parameter. Symbol must not have been previously defined
    void defineParameter(const char* symbol, double value);
    // Defines a new symbolic parameter. Symbol must not have been previously defined
    void defineParameter(const std::string& symbol, double value);
    // Defines new symbolic parameters. Symbols must not have been previously defined
    void defineParameters(const ModelParameters& inParams);

    // Updates a symbolic parameter value. The symbol must have been previously defined
    void updateParameter(const char* symbol, double newValue);
    // Updates a symbolic parameter value. The symbol must have been previously defined
    void updateParameter(const std::string& symbol, double newValue);
    /* Updates model parameter values.
    * NB: This never removes parameter definitions even if the input contains less parameters than what have been defined for this model */
    void updateParameters(const ModelParameters& newValues);

    /* Call to fix model contents (parameters and particles).
    * After locking, it is not possible to define new parameters or particles,
    * but calls to updateParameter() and similar are still valid.
    * MUST be called before readMatrixElements or similar.
    * Locking is required because runtime modifications to model definitions could immediately invalidate
    * dependent objects such as CollisionTensors, and we want to avoid hard-to-diagnoze bugs related to this. */
    void lockModelDefinitions();

    inline bool isLocked() const { return bLocked; }

    /* Read matrix elements from a file and stores them internally.
    This will only consider expressions where at least one currently registered out-of-equilibrium particle appears as an external particle.
    Note that this function clears any previously stored matrix elements.
    Returns false if something goes wrong. */
    bool readMatrixElements(
        const std::filesystem::path& matrixElementFile,
        bool bPrintMatrixElements = false);

    // Get copy of our cached matrix elements, grouped by out-of-equilibrium particle indices
    std::map<IndexPair, std::vector<MatrixElement>> getMatrixElements() { return mMatrixElements; }

    /* Creates a CollisionTensor object that includes all registered out-of-equilibrium particles.
    This uses stored matrix elements that should be setup with readMatrixElements() prior to calling this function. */
    CollisionTensor createCollisionTensor(size_t basisSize);

    PhysicsModel() {}
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
    std::vector<uint32_t> mOffEqIndices;

    // Cached matrix elements. This is std::map instead of std::unordered_map because we haven't defined a hash for IndexPair
    std::map<IndexPair, std::vector<MatrixElement>> mMatrixElements;

    // Must be called after changing mParameters to propagate changes to cached particle masses
    void updateParticleMassCache();

    void printMatrixElements() const;

    // ---- We use a very simple "observer pattern" to sync CollisionTensor objects when model parameters change
    std::vector<IModelObserver*> mObservers;
    
    void registerObserver(IModelObserver* observer);
    void unregisterObserver(IModelObserver* observer);
    void notifyModelChange(const ModelChangeContext& context) const;

    // ---- Factory-like functions for setupping collision integrals. Consider moving these to dedicated factory class(es) if the model becomes too 

    // Creates a CollisionTensor from cached matrix elements. Includes only off-eq particles specified in the input
    CollisionTensor createCollisionTensor(
        size_t basisSize,
        const std::vector<uint32_t>& offEqParticleIndices
    ) const;

    // Creates new CollisionIntegral4 for an off-eq particle pair. Matrix elements are read from matrixElementFile.
    CollisionIntegral4 createCollisionIntegral4(
        size_t basisSize,
        const IndexPair& offEqIndices) const;

    /* Create collision element for an off - eq pair. This will pass raw ParticleSpecies pointers to the resulting object. */
    CollisionElement<4> createCollisionElement(
        const IndexPair& offEqIndices,
        const MatrixElement& matrixElement) const;

    bool bLocked = false;
};



} // namespace
