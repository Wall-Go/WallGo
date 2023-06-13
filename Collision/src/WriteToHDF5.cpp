#include <iostream>

#include <cmath>
#include <vector>

#include "hdf5Interface.h"



// Write 4D array (vector) of doubles(!!) to .hdf5 file.
// Note that currently this will overwrite the whole file!

// Turn Array4D into C-style array, then pass that to overloaded WriteToHDF5()
void WriteToHDF5(const Array4D &data, std::string filename, std::string datasetName) {

   // 4D array dimensions
   hsize_t dims[4] = {data.size(), data[0].size(), data[0][0].size(), data[0][0][0].size()};

   // For writing we need to pass the data as a contiguous block of memory. A nested std::vector is not
   // guaranteed to be contiguous so need to flatten the data here
   std::vector<double> flattened_data;
   for (const auto& vec3 : data) {
      for (const auto& vec2 : vec3) {
         for (const auto& vec1 : vec2) {
               flattened_data.insert(flattened_data.end(), vec1.begin(), vec1.end());
         }
      }
   }

   unsigned int arrayDimension = 4;
   // need to pass C-style array
   WriteToHDF5(&flattened_data[0], arrayDimension, dims, filename, datasetName);
}


// arrayDimension = ie. is the array 3D or 4D or... 
// dims = data array dimensions
void WriteToHDF5(const double* data, unsigned int arrayDimension, const hsize_t* dims, 
                     std::string filename, std::string datasetName) {

   hid_t fileId, dataspaceId, datasetId;  

   int polynomialBasisSize = dims[0];
   (void)polynomialBasisSize; // suppress -Wunused-parameter

   // Create a new HDF5 file
   fileId = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
   if (fileId < 0) {
      std::cout << "Failed to create file '" << filename << "'. Exiting...\n";
      return;
   }

   // Create 4D dataspace for the dataset. Last argument NULL means the dataspace size is fixed
   dataspaceId = H5Screate_simple(4, dims, NULL);
   if (dataspaceId < 0) {
      printf("Failed to create dataspace. Exiting...\n");
      H5Fclose(fileId);
      return;
   }

   // Create a dataset of doubles (should work on all platforms)
   datasetId = H5Dcreate2(fileId, datasetName.c_str(), H5T_NATIVE_DOUBLE, dataspaceId,
                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   if (datasetId < 0) {
      printf("Failed to create dataset. Exiting...\n");
      H5Sclose(dataspaceId);
      H5Fclose(fileId);
      return;
   }

   // Write the data to the dataset. Note that we need to pass a C-array and not a std::vector
   herr_t status = H5Dwrite(datasetId, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
   // herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, wtf);
   if (status < 0) {
      printf("Failed to write data to dataset. Exiting...\n");
      H5Dclose(datasetId);
      H5Sclose(dataspaceId);
      H5Fclose(fileId);
      return;
   }

   // Attribute for storing a string ("Chebyshev" or "Cardinal")
   // TODO write the attribute at root level

   // Create an attribute dataspace for storing metadata, eg. which polynomial basis was used.
   // For now a dataspace of dimension 1, length 1 will do
   hsize_t attributeDims[1] = {1};
   hid_t attributeDataspaceId = H5Screate_simple(1, attributeDims, NULL);

   // TODO write accuracy, list of heavy/light particles etc

   std::string label = "Basis";
   std::string basisType = "Chebyshev";
   
   // Copy the built-in H5 string type and resize it accordingly
   hid_t stringAttributeTypeId = H5Tcopy(H5T_C_S1);
   H5Tset_size(stringAttributeTypeId, basisType.length());
   
   // Now create the string attribute
   hid_t stringAttributeId = H5Acreate2(datasetId, label.c_str(), stringAttributeTypeId, attributeDataspaceId, H5P_DEFAULT, H5P_DEFAULT);

   // Write the string attribute value
   H5Awrite(stringAttributeId, stringAttributeTypeId, basisType.c_str());

   // Close the string attribute and string attribute dataspace
   H5Aclose(stringAttributeId);
   H5Tclose(stringAttributeTypeId);
   H5Sclose(attributeDataspaceId);

   // End attribute

   // Close all opened HDF5 objects
   H5Dclose(datasetId);
   H5Sclose(dataspaceId);
   H5Fclose(fileId);

   printf("Wrote data to file '%s'.\n", filename.c_str());
}


void testHDF5() {

   const int gridSizeN = 20;
   std::string filename = "collisions_Chebyshev_" + std::to_string(gridSizeN) + ".hdf5";

   // 20x20x20x20 std::vector, all initialized to 0.0
   Array4D collisionIntegrals(gridSizeN, gridSizeN, gridSizeN, gridSizeN, 0.0);

   // Fill with dummy data
   long seed = 13424213;
   srand48(seed);

   for (int i = 0; i < gridSizeN; ++i) {
      for (int j = 0; j < gridSizeN; ++j) {
         for (int k = 0; k < gridSizeN; ++k) {
            for (int l = 0; l < gridSizeN; ++l) {
               collisionIntegrals[i][j][k][l] = drand48();
            }
         }
      }
   }

   // TODO Oli's preferred notation was C(j,k ; m,n) with j = p_z, k = p_par, m = Tbar, n = Ttilde
   
   WriteToHDF5(collisionIntegrals, filename, "top");
}

