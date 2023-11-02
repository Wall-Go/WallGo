#ifndef HDF5_INTERFACE_H_
#define HDF5_INTERFACE_H_


#include <string>
#include <vector>

// C++ API for HDF5 files
#include <H5Cpp.h>

// Recursive boilerplate for D-dimensional std::vectors (I don't want to write vector<vector<vector<....)
template<int D, typename T>
struct Vec : public std::vector<Vec<D - 1, T>> {
	static_assert(D >= 1, "Vector dimension needs to be > 0");
	template<typename... Args>
	Vec(int n = 0, Args... args) : std::vector<Vec<D - 1, T>>(n, Vec<D - 1, T>(args...)) {
	}
};

template<typename T>
struct Vec<1, T> : public std::vector<T> {
	Vec(int n = 0, const T& val = T()) : std::vector<T>(n, val) {
	}
};

using Array4D = Vec<4, double>;


// Struct for holding metadata about collision tensor. Default values are set to prevent exceptions in HDF5 routines
struct H5Metadata {
	int basisSize = 1;
	std::string basisName = "Unknown";
	std::string integrator = "Unknown";
};


/* Write C-style array (contiguous memory) to HDF5 file.
arrayDimension = is the array 3D or 4D or etc. dims = data array dimensions. */
void writeDataSet(H5::H5File &h5File, const double* data, size_t arrayDimension, const hsize_t* dims, std::string datasetName);

// Write Array4D to HDF5 file. This will just flatten the Array4D into a C-style array and call the above WriteToHDF5() overload
void writeDataSet(H5::H5File &h5File, const Array4D &data, std::string datasetName);

// Write metadata struct to open H5 file. This is done using HDF5 attributes
void writeMetadata(H5::H5File &h5File, const H5Metadata &metadata);

// Test function with dummy output .hdf5 file
void testHDF5();

#endif // header guard