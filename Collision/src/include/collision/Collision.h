#ifndef COLLISION_H
#define COLLISION_H

#include <vector>
#include <map>
#include <string>
#include <chrono>

#include "CollElem.h"
#include "ParticleSpecies.h"
#include "CollisionIntegral.h"
#include "hdf5Interface.h"


using Array6D = Vec<6, double>;

/* Control class for carrying out the full computation of
* 2 -> 2 collision terms 
*/
class Collision {

public: 
    Collision(uint basisSize);

    // how many basis polynomials
    const uint basisSizeN;

    void addParticle(const ParticleSpecies& particle);
    void addCoupling(double coupling);

    // Calculates all integrals. Call only after settings particles and couplings
    void calculateCollisionIntegrals();

    /* Creates all collision elements that mix two out-of-eq particles (can be the same particle) 
    @todo The matrix elements are read from file. */
    std::vector<CollElem<4>> makeCollisionElements(const std::string &particleName1, const std::string &particleName2);

    /* Turns a symbolic string expression into usable CollElem<4>. 
    Our matrix elements are M[a,b,c,d] -> expr, here indices are the abcd identifiers for outgoing particles.
    This needs the off-eq particle 2 to set deltaF flags properly. particleName1 is not needed (could be inferred from indices[0]) */  
    CollElem<4> makeCollisionElement(const std::string &particleName2, const std::vector<uint> &indices, const std::string &expr);

    
    // Count how many independent collision integrals we have for N basis polynomials and M out-of-equilibrium particles. Will be of order N^4 * M^2
    static long countIndependentIntegrals(uint basisSize, uint outOfEqCount);

    // Calculate CollisionIntegral4 everywhere on the grid. Results are stored in the input arrays 
    void evaluateCollisionTensor(CollisionIntegral4 &collisionIntegral, Array4D& results, Array4D& errors);



protected:

    // Used to interrupt long-running functions. The python module will override this with its own checks
    virtual inline bool shouldContinueEvaluation() { return true; };

    // Prints stuff about how many integrals we've computed and ETC
    void reportProgress();

    // Populates the particleIndex map 
    void makeParticleIndexMap();

    // Checks which particles in our current 'particles' array are out-of-eq, then stores those in outOfEqParticles
    void findOutOfEquilibriumParticles();

    // List of all particles that contribute to collisions
    std::vector<ParticleSpecies> particles;

    // Masses of the above particles in a vector form. Same ordering. This is vacuum + thermal
    std::vector<double> massSquares;

    // List of Lagrangian parameters
    std::vector<double> couplings;

    // Mapping: particle name -> tensor index. Need to put out-of-eq particles first if the input doesn't have this @todo
    std::map<std::string, uint> particleIndex;

    // Failsafe flag so that we don't do anything stupid (hopefully)
    bool bMatrixElementsDone = false;

    // List of out-of-equilibrium particles, handled internally. @todo should be list of references, not objects?
    std::vector<ParticleSpecies> outOfEqParticles;

private:

    // Progress tracking 
    int computedIntegralCount = 0;
    int totalIntegralCount;
    // Initial progress check and time estimate after this many integrals (in one thread)
    int initialProgressInterval = 10;
    bool bFinishedInitialProgressCheck = false;
    std::chrono::steady_clock::time_point startTime;
    std::chrono::duration<double> elapsedTime;
};


#endif // header guard