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
#include "ModelChangeContext.h"

namespace wallgo
{

class IModelObserver;

/* Helper class for specifying model contents. This is separate from the actual PhysicsModel class because we don't want to allow model structure to change after creation */
class ModelDefinition
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

private:

    ModelParameters mParameters;
    std::vector<ParticleDescription> mParticleDescriptions;

    friend class PhysicsModel;
};

class PhysicsModel
{
public:

    // Construct a new PhysicsModel
    PhysicsModel(const ModelDefinition& modelDefinition);

    /* Rule of Three stuff to notify observers on model destruction.
    Note that ideally we'd avoid these by requiring that models are kept alive for longer than observers,
    but this can be hard especially in the Python module if we let Python GC take ownership of our objects. */
    
    ~PhysicsModel();
    // Copying a model will NOT make observers listen to the copied model
    PhysicsModel(const PhysicsModel& other);
    PhysicsModel& operator=(const PhysicsModel& other);

    /*
    // TODO should we move observer registrations?
    PhysicsModel(PhysicsModel&&) noexcept = default;
    PhysicsModel& operator=(PhysicsModel&&) noexcept = default;
    */

    // Updates a symbolic parameter value. The symbol must have been defined at model creation time
    void updateParameter(const char* symbol, double newValue);
    // Updates a symbolic parameter value. The symbol must have been defined at model creation time
    void updateParameter(const std::string& symbol, double newValue);
    /* Updates model parameter values.
    * NB: This never removes parameter definitions even if the input contains less parameters than what have been defined for this model */
    void updateParameters(const ModelParameters& newValues);

    /* Read matrix elements from a file and stores them internally.
    This will only consider expressions where at least one currently registered out-of-equilibrium particle appears as an external particle.
    Note that this function clears any previously stored matrix elements.
    Returns false if something goes wrong. */
    bool readMatrixElements(
        const std::filesystem::path& matrixElementFile,
        bool bPrintMatrixElements = false);

    // Get copy of our cached matrix elements, grouped by out-of-equilibrium particle indices
    std::map<IndexPair, std::vector<MatrixElement>> getMatrixElements() { return mMatrixElements; }

    void printMatrixElements() const;

    /* Creates a CollisionTensor object that includes all registered out-of-equilibrium particles, and registers it as a model observer.
    This uses stored matrix elements that should be setup with readMatrixElements() prior to calling this function. */
    CollisionTensor createCollisionTensor(size_t basisSize);

    // ---- We use a simple "observer pattern" to sync CollisionTensor objects when model parameters change
    void registerObserver(IModelObserver& observer);
    void unregisterObserver(IModelObserver& observer);
    size_t getNumObservers() const { return mObservers.size(); }

private:

    ModelParameters mParameters;
    
    /* Use custom indices for particle lookup. Will send copies of ParticleSpecies objects to collision elements. */
    std::unordered_map<uint32_t, ParticleSpecies> mParticles;

    ParticleNameMap mParticleNameMap;
    // Indices of out-of-equilibrium particles
    std::vector<uint32_t> mOffEqIndices;

    // Cached matrix elements. This is std::map instead of std::unordered_map because we haven't defined a hash for IndexPair
    std::map<IndexPair, std::vector<MatrixElement>> mMatrixElements;

    /* Called after changing mParameters to figure out how particle properties change. Currently: 
        1) The only property that can change is the cached mass
        2) We do the calculation for all particles even if some of them remain unchanged.
        This is because we don't have logic in place to figure out which particles depend on which params.
    */
    std::vector<ParticleChangeContext> computeParticleChanges();

    std::vector<IModelObserver*> mObservers;
    void notifyModelChange(const ModelChangeContext& context) const;

    // ---- Factory-like functions for setupping collision integrals. Consider moving these to dedicated factory class(es) if the model becomes too 

    // Creates a CollisionTensor from cached matrix elements and registers it as a model observer. Includes only off-eq particles specified in the input
    CollisionTensor createCollisionTensor(
        size_t basisSize,
        const std::vector<uint32_t>& offEqParticleIndices
    );

    // Creates new CollisionIntegral4 for an off-eq particle pair. Matrix elements are read from matrixElementFile.
    CollisionIntegral4 createCollisionIntegral4(
        size_t basisSize,
        const IndexPair& offEqIndices) const;

    /* Create collision element for an off-eq pair. This will pass raw ParticleSpecies pointers to the resulting object. */
    CollisionElement<4> createCollisionElement(
        const IndexPair& offEqIndices,
        const MatrixElement& matrixElement) const;
};



} // namespace
