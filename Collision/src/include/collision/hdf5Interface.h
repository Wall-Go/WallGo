#ifndef HDF5_INTERFACE_H_
#define HDF5_INTERFACE_H_


#include <string>
#include <vector>

// TODO proper include path. Dunno why the header is in this weird location
// #include "/usr/include/hdf5/serial/hdf5.h"
#include "hdf5.h"

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


// Turn Array4D into C-style array, then pass that to overloaded WriteToHDF5()
void WriteToHDF5(const Array4D &data, std::string filename, std::string datasetName);
void WriteToHDF5(const double* data, unsigned int arrayDimension, const hsize_t* dims, std::string filename, std::string datasetName);

#endif // header guard