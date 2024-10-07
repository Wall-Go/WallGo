#pragma once

#include <string>
#include <vector>
#include <H5Cpp.h> // C++ API for HDF5 files

#include "EnvironmentMacros.h"
#include "Utils.h"

namespace wallgo
{

struct CollisionMetadata;

namespace utils
{

/* Write C-style array (contiguous memory) to HDF5 file.
arrayDimension = is the array 3D or 4D or etc. dims = data array dimensions. */
void writeDataSet(H5::H5File &h5File, const double* data, size_t arrayDimension, const hsize_t* dims, std::string datasetName);

// Write Array4D to HDF5 file. This will just flatten the Array4D into a C-style array and call the above WriteToHDF5() overload
void writeDataSet(H5::H5File &h5File, const Array4D &data, std::string datasetName);

// Write metadata struct to open H5 file. This is done using HDF5 attributes
void writeMetadata(H5::H5File &h5File, const CollisionMetadata& metadata);


} // namespace utils
} // namespace wallgo
