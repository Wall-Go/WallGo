#pragma once

#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <filesystem>
#include <utility> // std::pair
#include <memory>
#include <cstdint>

#include "Common.h"
#include "ModelParameters.h"
#include "CollisionElement.h"
#include "ParticleSpecies.h"
#include "CollisionIntegral.h"
#include "ResultContainers.h"


/** How we manage particles. Calling addParticle(particle) registers a new particle with the CollisionTensor.
 * We take a copy of the particle struct and store them as unique pointers in our 'particles' list.
 * Each CollisionElement needs raw pointers to its external particles, so when new CollElems are created through the manager
 * we pass references to appropriate particles from our 'particles' list.
 */

/** Handling of model parameters. The manager holds a map of [string, double] pairs that can be updated
 * through CollisionTensor::setVariable(key, val). These need to be defined before parsing matrix elements,
 * otherwise the parsing will error out due to undefined symbols. Each matrix element will hold their own internal map
 * of parameter values, so if they change at any point, it is up to the manager to sync all built CollisionIntegral objects
 * and their stored MatrixElement objects. 
 * 
 * FIXME: Would it be better to pass the parameters to matrix elements as pointers too? 
*/

namespace wallgo
{

class PhysicsModel;

/* CollisionTensor is the main interface to computing WallGo collision integrals. 
* Manages model-parameter and particle definitions, and construct collision integral objects
* based on matrix element input. Used also to initiate collision integrations. */
class WALLGO_API CollisionTensor 
{

public: 

    CollisionTensor(const PhysicsModel* creator);
    CollisionTensor(const PhysicsModel* creator, size_t basisSize);

    /* Configures default integration options that are used by computeIntegralsForPair() and related functions
    if no IntegrationOptions object is passed when calling them. */
    void setDefaultIntegrationOptions(const IntegrationOptions& options);

    /* Configures default integration verbosity that is used by computeIntegralsForPair() and related functions
    if no CollisionIntegralVerbosity object is passed when calling them. */
    void setDefaultIntegrationVerbosity(const CollisionTensorVerbosity& verbosity);

    // Change basis size used by the polynomial grid. This is a fast operation, does not require rebuild of stored integral objects
    void changePolynomialBasisSize(size_t newBasisSize);

    /* Adds a new collision integral object for an off - eq particle pair.
    Currently we always have just one CollisionIntegral4 for each pair, so this will override any existing integral for that pair.
    NB: does NOT perform sensibility checks on the input integral.*/
    void addCollisionIntegral(const ParticleNamePair& particleNames, const CollisionIntegral4& inIntegral);

    // ---- Evaluating cached integrals

    /* Calculate collision integrals for particle pair (particle1, particle2) everywhere on the grid. */
    CollisionResultsGrid computeIntegralsForPair(
        const std::string& particle1,
        const std::string& particle2,
        const IntegrationOptions& options,
        const CollisionTensorVerbosity& verbosity);

    /* Calculate collision integrals for particle pair (particle1, particle2) everywhere on the grid.
    Uses default integration options. */
    CollisionResultsGrid computeIntegralsForPair(
        const std::string& particle1,
        const std::string& particle2,
        const CollisionTensorVerbosity& verbosity);

    /* Calculate collision integrals for particle pair (particle1, particle2) everywhere on the grid.
    Uses default verbosity settings */
    CollisionResultsGrid computeIntegralsForPair(
        const std::string& particle1,
        const std::string& particle2,
        const IntegrationOptions& options);


    /* Calculate collision integrals for particle pair (particle1, particle2) everywhere on the grid.
    Uses default integration options and verbosity settings. */
    CollisionResultsGrid computeIntegralsForPair(
        const std::string& particle1,
        const std::string& particle2);

    /* Calculates all integrals previously initialized with setupCollisionIntegrals(). */
    CollisionTensorResult computeIntegralsAll(const IntegrationOptions& options, const CollisionTensorVerbosity& verbosity);

    /* Calculates all integrals previously initialized with setupCollisionIntegrals().
    Uses default verbosity. */
    CollisionTensorResult computeIntegralsAll(const IntegrationOptions& options);

    /* Calculates all integrals previously initialized with setupCollisionIntegrals().
    Uses default options. */
    CollisionTensorResult computeIntegralsAll(const CollisionTensorVerbosity& verbosity);

    /* Calculates all integrals previously initialized with setupCollisionIntegrals().
    Uses default options and verbosity. */
    CollisionTensorResult computeIntegralsAll();

    // Count how many independent collision integrals we have. Scales as N^4 * M^2, N = grid size, M = number of off-eq particles
    size_t countIndependentIntegrals() const;

    /* Used to sync matrix elements etc with changes to model parameters */
    void updateModelParameters(const ModelParameters& changedParameters);

private:

    const PhysicsModel* mModel = nullptr;

    size_t mBasisSize;

    IntegrationOptions mDefaultIntegrationOptions;
    CollisionTensorVerbosity mDefaultVerbosity;

    // Holds collision integrals so that they can be reused
    std::map<ParticleNamePair, CollisionIntegral4> mCachedIntegrals;
};

} // namespace
