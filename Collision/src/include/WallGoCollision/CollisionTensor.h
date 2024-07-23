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

/* CollisionTensor is the main interface to computing WallGo collision integrals. 
* Manages model-parameter and particle definitions, and construct collision integral objects
* based on matrix element input. Used also to initiate collision integrations. */
class WALLGO_API CollisionTensor 
{

public: 
    CollisionTensor();
    CollisionTensor(size_t basisSize);

    /* Configures default integration options that are used by computeIntegralsForPair() and related functions
    if no IntegrationOptions object is passed when calling them. */
    void setDefaultIntegrationOptions(const IntegrationOptions& options);

    /* Configures default integration verbosity that is used by computeIntegralsForPair() and related functions
    if no CollisionIntegralVerbosity object is passed when calling them. */
    void setDefaultIntegrationVerbosity(const CollisionTensorVerbosity& verbosity);

    // Change basis size used by the polynomial grid. This is a fast operation, does not require rebuild of stored integral objects
    void changePolynomialBasisSize(size_t newBasisSize);


    // Creates new CollisionIntegral4 for an off-eq particle pair. Matrix elements are read from matrixElementFile.
    CollisionIntegral4 setupCollisionIntegral(
        const std::shared_ptr<ParticleSpecies>& particle1,
        const std::shared_ptr<ParticleSpecies>& particle2, 
        const std::string &matrixElementFile,
        size_t inBasisSize,
        bool bVerbose = false);

    /* Initializes and caches collision integrals for all registered particles. Basis size and matrix element file need to be set before calling this.
    Note that calling this will clear any previously stored collision integral objects. */
    void setupCollisionIntegrals(const std::filesystem::path& matrixElementFile, bool bVerbose = false);
    
    // Clears all stored collision integral objects
    void clearIntegralCache();

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

protected:

    /* Turns a symbolic string expression into usable CollisionElement<4>. 
    Our matrix elements are M[a,b,c,d] -> expr, here indices are the abcd identifiers for outgoing particles.
    This needs the off-eq particle 2 to set deltaF flags properly. particleName1 is not needed (could be inferred from indices[0]). 
    The symbols array needs to contain all free symbols that appear in the expr, apart from 's','t','u' which are automatically defined.
    As a sensibility check, the symbols MUST be contained in our modelParameters map, from which we also pick initial values for the symbols. */  
    CollisionElement<4> makeCollisionElement(const std::string &particleName2, const std::vector<size_t> &indices,
        const std::string &expr, const std::vector<std::string>& symbols);
    
private:

    size_t mBasisSize = 1;

    IntegrationOptions mDefaultIntegrationOptions;
    CollisionTensorVerbosity mDefaultVerbosity;

    // Holds collision integrals so that they can be reused
    std::map<ParticleNamePair, CollisionIntegral4> mCachedIntegrals;
};

} // namespace
