#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include "Common.h"
#include "hdf5Interface.h"

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
struct CollisionMetadata
{
    size_t basisSize = 1;
    bool bStatisticalErrors = false;
    std::string basisName = "Unknown";
    std::string integrator = "Unknown";
};

// Rank 4 tensor that holds collision integration results on the grid for (particle1, particle2) pair
class WALLGO_API CollisionResultsGrid
{
public:

    CollisionResultsGrid() : mParticlePair("Unknown", "Unknown"), mElementsPerDimension(0) {}
    CollisionResultsGrid(const ParticleNamePair& particlePair, const CollisionMetadata& metadata);

    // Move semantics needed to handle the dynamic errors array
    CollisionResultsGrid(const CollisionResultsGrid& other) = delete;
    CollisionResultsGrid& operator=(const CollisionResultsGrid& other) = delete;
    CollisionResultsGrid(CollisionResultsGrid&& other) noexcept = default;
    CollisionResultsGrid& operator=(CollisionResultsGrid&& other) noexcept = default;

    inline bool hasStatisticalErrors() const { return mErrors != nullptr; }
    
    size_t getBasisSize() const { return mMetadata.basisSize; }
    ParticleNamePair getParticleNamePair() const { return mParticlePair; }

    // ---- Read-write accessors

    double valueAt(size_t m, size_t n, size_t j, size_t k) const;
    double& valueAt(size_t m, size_t n, size_t j, size_t k);

    double errorAt(size_t m, size_t n, size_t j, size_t k) const;
    double& errorAt(size_t m, size_t n, size_t j, size_t k);

    // Write (value, error) pair. Error will only be written if the CollisionResultsGrid object was created with bIncludeStatisticalErrors = true
    void updateValue(size_t m, size_t n, size_t j, size_t k, double newValue, double newError);

    /* Write array contents to a HDF5 file. This always overrides file if it exists.
    Dataset name for the integration results will be "particle1, particle2" with particle names read from mDescription.
    Dataset name for integration errors will be "particle1, particle2 errors"
    Return value is false if something goes wrong. */
    bool writeToHDF5(
        const std::filesystem::path& filePath,
        bool bWriteErrors = true) const;

private:

    ParticleNamePair mParticlePair;
    
    size_t mElementsPerDimension;  // equals basisSize - 1

    Array4D mData;
    // Optional: statistical errors of integrations. Has large memory cost, so we allocate this only if needed
    std::unique_ptr<Array4D> mErrors = nullptr;

    CollisionMetadata mMetadata;
    void initData();
};

// TODO replace the manager with CollisionTensor

struct WALLGO_API CollisionTensorResult
{
public:

    CollisionTensorResult() {}
    CollisionTensorResult(size_t numParticlePairs) : mData(numParticlePairs) {}

    /* Writes the contents to disk, so that each particle pair is written into a separate HDF5 file.
    * In practice this just calls CollisionResultsGrid::writeToHDF5 for each particle pair.
    * Return value is false if something went wrong. */
    bool writeToIndividualHDF5(
        const std::filesystem::path& outDirectory,
        bool bWriteErrors = true) const;

    /* Writes the contents to current working directory, so that each particle pair is written into a separate HDF5 file.
    * In practice this just calls CollisionResultsGrid::writeToHDF5 for each particle pair.
    * Return value is false if something went wrong. */
    bool writeToIndividualHDF5(bool bWriteErrors = true) const;

    // Returns pointer to collision integration results of the specified particle pair. Can be nullptr if the pair is not found
    CollisionResultsGrid* getResultsForParticlePair(const ParticleNamePair& particlePair);

    // Returns pointer to collision integration results of the specified particle pair. Can be nullptr if the pair is not found
    CollisionResultsGrid* getResultsForParticlePair(const std::string& particleName1, const std::string& particleName2);

    // CollisionResultsGrid is not copyable, so use std::vector
    std::vector<CollisionResultsGrid> mData;
};

} // namespace
