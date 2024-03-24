#ifndef COLLISION_H
#define COLLISION_H

#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <filesystem>
#include <utility> // std::pair
#include <memory>

#include "CollElem.h"
#include "ParticleSpecies.h"
#include "CollisionIntegral.h"
#include "hdf5Interface.h"


namespace wallgo
{

/* Control class for carrying out the full computation of
* 2 -> 2 collision terms */
class CollisionManager 
{

public: 
    CollisionManager();
    CollisionManager(size_t basisSize);

    void addParticle(const ParticleSpecies& particle);

    // Change basis size used by the polynomial grid. Does not force rebuild of stored integral objects
    void changePolynomialBasis(size_t newBasisSize);

    // Set numerical value to a physics parameter used in matrix elements. Registers a new variable if the name is not already defined. 
    void setVariable(const std::string& name, double value);

    // Creates new CollisionIntegral4 for an off-eq particle pair. Matrix elements are read from matrixElementFile.
    CollisionIntegral4 setupCollisionIntegral(const std::shared_ptr<ParticleSpecies>& particle1, const std::shared_ptr<ParticleSpecies>& particle2, 
        const std::string &matrixElementFile, size_t basisSize, bool bVerbose = false);

    /* Initializes and caches collision integrals for all registered particles. Basis size and matrix element file need to be set before calling this.
    Note that calling this will clear any previously stored collision integral objects.*/
    void setupCollisionIntegrals(bool bVerbose = false);
    
    void clearCollisionIntegrals();
    
    /* Calculate CollisionIntegral4 everywhere on the grid. Results are stored in the input arrays. 
    Integration options are read from out internal integrationOptions struct. */ 
    void evaluateCollisionTensor(CollisionIntegral4 &collisionIntegral, 
        Array4D& results, Array4D& errors, bool bVerbose = false);

    /* Calculates all integrals previously initialized with setupCollisionIntegrals().
    Options for the integration can be changed by with CollisionManager::configureIntegration().
    If bVerbose is true, will print each result to stdout. */
    void calculateCollisionIntegrals(bool bVerbose = false);

    // Count how many independent collision integrals we have for N basis polynomials and M out-of-equilibrium particles. Will be of order N^4 * M^2
    static long countIndependentIntegrals(size_t basisSize, size_t outOfEqCount);


    void configureIntegration(const IntegrationOptions& options);

    // Specify where to store output files, relative or absolute path. Defaults to current work directory.
    void setOutputDirectory(const std::string& directoryName);

    /* Specify file to read matrix elements from, relative or absolute path. Default is "MatrixElements.txt". 
    Return value is false if the file was not found, true otherwise. */
    bool setMatrixElementFile(const std::string& filePath);

protected:

    // Used to interrupt long-running functions. The python module will override this with its own checks
    virtual inline bool shouldContinueEvaluation() { return true; };

    // Prints stuff about how many integrals we've computed and ETC
    void reportProgress();

    bool particleRegistered(const ParticleSpecies& particle);

    /* Turns a symbolic string expression into usable CollElem<4>. 
    Our matrix elements are M[a,b,c,d] -> expr, here indices are the abcd identifiers for outgoing particles.
    This needs the off-eq particle 2 to set deltaF flags properly. particleName1 is not needed (could be inferred from indices[0]). 
    The symbols array needs to contain all free symbols that appear in the expr, apart from 's','t','u' which are automatically defined.
    As a sensibility check, the symbols MUST be contained in our modelParameters map, from which we also pick initial values for the symbols. */  
    CollElem<4> makeCollisionElement(const std::string &particleName2, const std::vector<size_t> &indices,
        const std::string &expr, const std::vector<std::string>& symbols);
    
    // Creates all collision elements that mix two out-of-eq particles (can be the same particle).
    std::vector<CollElem<4>> parseMatrixElements(const std::string &particleName1, const std::string &particleName2,
        const std::string &matrixElementFile, bool bVerbose = false);


    /* Holds collision integrals so that they can be reused. Keys are pairs of particle names. */
    std::map<std::pair<std::string, std::string>, CollisionIntegral4> integrals;

    // TODO have some way of preventing duplicate particles in the above lists

    std::map<std::string, double> modelParameters;

private:

    size_t basisSize = 0;

    // Progress tracking 
    int computedIntegralCount = 0;
    int totalIntegralCount;
    // Initial progress check and time estimate after this many integrals (in one thread)
    int initialProgressInterval = 10;
    bool bFinishedInitialProgressCheck = false;
    std::chrono::steady_clock::time_point startTime;
    std::chrono::duration<double> elapsedTime;

    IntegrationOptions integrationOptions;
    std::filesystem::path outputDirectory;
    std::filesystem::path matrixElementFile;

    /** How we manage particles. When new particle is registered through addParticle(particle), we store a copy of that particle
     * in our 'particles' list. For off-equilibrium particles, we also store a raw pointer to the added particle in 'outOfEqParticles' list.
     * CollElem objects need references to external particles in their respective matrix element. 
     */

    // List of all particles that contribute to collisions
    std::vector<std::shared_ptr<ParticleSpecies>> particles;

    // List of out-of-equilibrium particles, managed internally
    std::vector<std::shared_ptr<ParticleSpecies>> outOfEqParticles;

    // Mapping: particle name -> tensor index. Ordering does not matter.
    std::map<std::string, size_t> particleIndex;
};

} // namespace


#endif // header guard