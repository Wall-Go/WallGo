#include "CollisionTensorResult.h"
#include "hdf5Interface.h"

#include <cassert>
#include <iostream>

namespace wallgo
{
CollisionTensorResult::CollisionTensorResult(const CollisionTensorDesc& description)
    : mDescription(description)
{
    initData();
}

CollisionTensorResult::CollisionTensorResult(size_t basisSize, bool bIncludeStatisticalErrors)
{
    mDescription.basisSize = basisSize;
    mDescription.bStatisticalErrors = bIncludeStatisticalErrors;
    initData();
}

double CollisionTensorResult::valueAt(size_t m, size_t n, size_t j, size_t k) const
{
    assert(m < mElementsPerDimension && n < mElementsPerDimension && j < mElementsPerDimension && k < mElementsPerDimension);
    return mData[m][n][j][k];
}

double& CollisionTensorResult::valueAt(size_t m, size_t n, size_t j, size_t k)
{
    assert(m < mElementsPerDimension && n < mElementsPerDimension && j < mElementsPerDimension && k < mElementsPerDimension);
    return mData[m][n][j][k];
}

double CollisionTensorResult::errorAt(size_t m, size_t n, size_t j, size_t k) const
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

double& CollisionTensorResult::errorAt(size_t m, size_t n, size_t j, size_t k)
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

void CollisionTensorResult::updateValue(size_t m, size_t n, size_t j, size_t k, double newValue, double newError)
{
    valueAt(m, n, j, k) = newValue;
    if (hasStatisticalErrors())
    {
        errorAt(m, n, j, k) = newError;
    }
}

bool CollisionTensorResult::writeToHDF5(const std::filesystem::path& filePath, bool bWriteErrors)
{
    // H5F_ACC_TRUNC = overwrite existing file
    H5::H5File h5File(filePath.string(), H5F_ACC_TRUNC);

    utils::writeMetadata(h5File, mDescription);

    const std::string datasetName = mDescription.particle1 + ", " + mDescription.particle2;
    
    utils::writeDataSet(h5File, mData, datasetName);
    if (bWriteErrors && hasStatisticalErrors())
    {
        utils::writeDataSet(h5File, *mErrors, datasetName + " errors");
    }

    h5File.close();

    return true;
}

void CollisionTensorResult::initData()
{
    mElementsPerDimension = mDescription.basisSize - 1;
    if (mElementsPerDimension > 0)
    {
        mData = Array4D(mElementsPerDimension, mElementsPerDimension, mElementsPerDimension, mElementsPerDimension, 0.0);
        if (mDescription.bStatisticalErrors)
        {
            mErrors = std::make_unique<Array4D>(mElementsPerDimension, mElementsPerDimension, mElementsPerDimension, mElementsPerDimension, 0.0);
        }
    }
}


} // namespace