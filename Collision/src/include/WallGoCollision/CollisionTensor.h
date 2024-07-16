#ifndef COLLISIONTENSOR_H
#define COLLISIONTENSOR_H

#include "EnvironmentMacros.h"
#include "Utils.h"
#include "hdf5Interface.h"

#include <memory>

namespace wallgo
{

/** The linearized Boltzmann equations for deltaF (deviation from equilibrium) is of form
    (L_ab + T^2 C_ab) deltaF_b = S_a
where a,b label particle species, L is the Liouville operator and S is a source term.
C_ab is the (dimensionless) collision operator which we focus on here.

WallGo follows the spectral approach of 2204.13120 in which deltaF is expanded as a series of "basis polynomials" T_m.
The collision operator is rank-2 in particle indices (a,b), rank-2 in polynomial indices (m,n)
and rank-2 in momentum indices (j, k) (momentum components of particle 'a' that are parallel and perpendicular to the wall).
In total C is a rank-6 tensor, assuming the collision integrals are not explicitly position-dependent.

Note that the last assumption fails if the (dimensionless) C_ab contain explicit T-dependence,
which occurs eg. if vacuum masses are included in matrix elements. We do not consider this more complicated case.

Here we describe this as a 2D array of 4D objects so that particle indices are separated from the (polynomial, momentum) "grid" indices.
*/

// Rank-4 tensor that collects numerical collision data for (particle1, particle2) pair
class WALLGO_API CollisionTensor
{
public:
    CollisionTensor(size_t basisSize_, bool bIncludeStatisticalErrors);

    const size_t basisSize;

    inline bool hasStatisticalErrors() const { return errors != nullptr; }
    
    double valueAt(size_t m, size_t n, size_t j, size_t k) const;
    // Read-write access to the element at [m][n][j][k]
    double& valueAt(size_t m, size_t n, size_t j, size_t k);

    double errorAt(size_t m, size_t n, size_t j, size_t k) const;
    // Read-write access to the statistical error of element at [m][n][j][k]
    double& errorAt(size_t m, size_t n, size_t j, size_t k);

    CollisionTensor(const CollisionTensor& other);
    CollisionTensor& operator=(const CollisionTensor& other);
    CollisionTensor(CollisionTensor&& other) noexcept;
    CollisionTensor& operator=(CollisionTensor&& other) noexcept;
    
private:
    Array4D data;
    
    // Optional: statistical errors of integrations. Has large memory cost, so we allocate this only if needed
    std::unique_ptr<Array4D> errors = nullptr;
    
    const size_t elementsPerDimension;  // equals basisSize - 1
};

} // namespace
#endif