#include "ResultContainers.h"
#include "hdf5Interface.h"

#include <cassert>
#include <iostream>

namespace wallgo
{


CollisionResultsGrid::CollisionResultsGrid(const ParticleNamePair& particlePair, const CollisionMetadata& metadata)
    : mParticlePair(particlePair),
    mMetadata(metadata)
{
    initData();
}

double CollisionResultsGrid::valueAt(size_t m, size_t n, size_t j, size_t k) const
{
    assert(m < mElementsPerDimension && n < mElementsPerDimension && j < mElementsPerDimension && k < mElementsPerDimension);
    return mData[m][n][j][k];
}

double& CollisionResultsGrid::valueAt(size_t m, size_t n, size_t j, size_t k)
{
    assert(m < mElementsPerDimension && n < mElementsPerDimension && j < mElementsPerDimension && k < mElementsPerDimension);
    return mData[m][n][j][k];
}

double CollisionResultsGrid::errorAt(size_t m, size_t n, size_t j, size_t k) const
{
    assert(m < mElementsPerDimension && n < mElementsPerDimension && j < mElementsPerDimension && k < mElementsPerDimension);

    if (!hasStatisticalErrors())
    {
        std::cerr << "Warning: attempted to read statistical error of integral result at [m, n, j, k] = ["
            << m << ", " << n << ", " << j << ", " << k << "], but the integrals were calculated without storing errors" << std::endl;
        return 0.0;
    }

    return (*mErrors.get())[m][n][j][k];
}

double& CollisionResultsGrid::errorAt(size_t m, size_t n, size_t j, size_t k)
{
    assert(m < mElementsPerDimension && n < mElementsPerDimension && j < mElementsPerDimension && k < mElementsPerDimension);

    if (!hasStatisticalErrors())
    {
        std::cerr << "Warning: attempted writing to statistical error of integral result at [m, n, j, k] = ["
            << m << ", " << n << ", " << j << ", " << k << "], but the integrals were calculated without storing errors" << std::endl;
        
        // this is always fatal
        std::exit(444);
    }

    return (*mErrors.get())[m][n][j][k];
}

void CollisionResultsGrid::updateValue(size_t m, size_t n, size_t j, size_t k, double newValue, double newError)
{
    valueAt(m, n, j, k) = newValue;
    if (hasStatisticalErrors())
    {
        errorAt(m, n, j, k) = newError;
    }
}

bool CollisionResultsGrid::writeToHDF5(const std::filesystem::path& filePath, bool bWriteErrors) const
{
    // H5F_ACC_TRUNC = overwrite existing file
    H5::H5File h5File(filePath.string(), H5F_ACC_TRUNC);

    utils::writeMetadata(h5File, mMetadata);

    const std::string datasetName = mParticlePair.first + ", " + mParticlePair.second;
    
    utils::writeDataSet(h5File, mData, datasetName);
    if (bWriteErrors && hasStatisticalErrors())
    {
        utils::writeDataSet(h5File, *mErrors, datasetName + " errors");
    }

    h5File.close();

    return true;
}

void CollisionResultsGrid::initData()
{
    mElementsPerDimension = mMetadata.basisSize - 1;
    if (mElementsPerDimension > 0)
    {
        mData = Array4D(mElementsPerDimension, mElementsPerDimension, mElementsPerDimension, mElementsPerDimension, 0.0);
        if (mMetadata.bStatisticalErrors)
        {
            mErrors = std::make_unique<Array4D>(mElementsPerDimension, mElementsPerDimension, mElementsPerDimension, mElementsPerDimension, 0.0);
        }
    }
}


bool CollisionTensorResult::writeToIndividualHDF5(const std::filesystem::path& outDirectory, bool bWriteErrors) const
{
    namespace fs = std::filesystem;

    // Create the directory if it doesn't exist
    fs::path dir(outDirectory);
    if (!fs::exists(dir))
    {
        try
        {
            fs::create_directory(dir);
        }
        catch (const fs::filesystem_error& e)
        {
            std::cerr << "Failed to create collision output dir: " << dir.string()
                << ". Error was: " << e.what() << std::endl;
            return false;
        }
    }

    bool bAllOK = true;

    for (const CollisionResultsGrid& resultsForPair : mData)
    {
        const ParticleNamePair pair = resultsForPair.getParticleNamePair();
        const std::string fnameBase = "collisions_" + pair.first + "_" + pair.second + ".hdf5";
        const fs::path outFile = outDirectory / fnameBase;
        bAllOK &= resultsForPair.writeToHDF5(outFile, bWriteErrors);
    }

    return bAllOK;
}

bool CollisionTensorResult::writeToIndividualHDF5(bool bWriteErrors) const
{
    return writeToIndividualHDF5(std::filesystem::current_path(), bWriteErrors);
}

CollisionResultsGrid* CollisionTensorResult::getResultsForParticlePair(const ParticleNamePair& particlePair)
{
    for (size_t i = 0; i < mData.size(); ++i)
    {
        if (mData[i].getParticleNamePair() == particlePair)
        {
            return &mData[i];
        }
    }

    return nullptr;
}

CollisionResultsGrid* CollisionTensorResult::getResultsForParticlePair(const std::string& particleName1, const std::string& particleName2)
{
    return getResultsForParticlePair(ParticleNamePair(particleName1, particleName2));
}

} // namespace