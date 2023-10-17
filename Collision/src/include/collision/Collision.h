#ifndef COLLISION_H
#define COLLISION_H

#include <vector>
#include <map>
#include <string>

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

    CollElem<4> makeCollisionElement(const std::string &particleName1, const std::string &particleName2, const std::string &readMatrixElement);

    
    // Count how many independent collision integrals we have for N basis polynomials and M out-of-equilibrium particles. Will be of order N^4 * M^2
    static long countIndependentIntegrals(uint basisSize, uint outOfEqCount);

    // Calculate CollisionIntegral4 everywhere on the grid. Results are stored in the input arrays 
    void evaluateCollisionTensor(CollisionIntegral4 &collisionIntegral, Array4D& results, Array4D& errors);

private:

    void makeParticleIndexMap();
    void findOutOfEquilibriumParticles();

    // Processes string of form "M[a,b,c,d] -> some funct" and stores in the arguments
    void extractSymbolicMatrixElement(const std::string &inputString, std::vector<uint> &indices, std::string &mathExpression);

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

    // List of out-of-equilibrium particles, handled internally
    std::vector<ParticleSpecies> outOfEqParticles;


    // @todo config file for file paths 
    
    // Directory where we search for matrix elements
    std::string matrixElementDirectory = "MatrixElements";

    // Base file name for matrix element files. We append particle names to this like "_top_top"
    std::string matrixElementFileNameBase = "matrixElements";
};


#endif // header guard