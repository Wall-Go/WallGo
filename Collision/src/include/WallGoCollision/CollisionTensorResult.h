#ifndef COLLISIONTENSORRESULT_H
#define COLLISIONTENSORRESULT_H

#include "EnvironmentMacros.h"
#include "Utils.h"
#include "hdf5Interface.h"

#include <filesystem>
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


// Holds metadata about collision integrations
struct CollisionTensorDesc
{
    size_t basisSize = 1;
    bool bStatisticalErrors = true;
    std::string particle1 = "Unknown";
    std::string particle2 = "Unknown";
    std::string basisName = "Unknown";
    std::string integrator = "Unknown";
};

// Rank-4 tensor that collects numerical collision data for (particle1, particle2) pair
class WALLGO_API CollisionTensorResult
{
public:
    CollisionTensorResult(const CollisionTensorDesc& description);
    CollisionTensorResult(size_t basisSize, bool bIncludeStatisticalErrors);

    // Move semantics needed to handle the dynamic errors array
    CollisionTensorResult(const CollisionTensorResult& other) = delete;
    CollisionTensorResult& operator=(const CollisionTensorResult& other) = delete;
    CollisionTensorResult(CollisionTensorResult&& other) noexcept = default;
    CollisionTensorResult& operator=(CollisionTensorResult&& other) noexcept = default;

    inline bool hasStatisticalErrors() const { return mErrors != nullptr; }
    
    size_t getBasisSize() const { return mDescription.basisSize; }

    // ---- Read-write accessors

    double valueAt(size_t m, size_t n, size_t j, size_t k) const;
    double& valueAt(size_t m, size_t n, size_t j, size_t k);

    double errorAt(size_t m, size_t n, size_t j, size_t k) const;
    double& errorAt(size_t m, size_t n, size_t j, size_t k);

    // Write (value, error) pair. Error will only be written if the CollisionTensorResult object was created with bIncludeStatisticalErrors = true
    void updateValue(size_t m, size_t n, size_t j, size_t k, double newValue, double newError);

    /* Write array contents to a HDF5 file. This always overrides file if it exists.
    Dataset name for the integration results will be "particle1, particle2" with particle names read from mDescription.
    Dataset name for integration errors will be "particle1, particle2 errors"
    Return value is false if something goes wrong. */
    bool writeToHDF5(
        const std::filesystem::path& filePath,
        bool bWriteErrors = true);

private:
    Array4D mData;
    // Optional: statistical errors of integrations. Has large memory cost, so we allocate this only if needed
    std::unique_ptr<Array4D> mErrors = nullptr;

    CollisionTensorDesc mDescription;
    size_t mElementsPerDimension;  // equals basisSize - 1

    void initData();
};

} // namespace
#endif