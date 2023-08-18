#include <iostream>

#include <cmath>
#include <vector>

#include "hdf5Interface.h"


void writeMetadata(H5::H5File &h5File, const H5Metadata &metadata) {

	try {

		// Create a group to hold metadata (keeping it separate from the actual data)
		H5::Group metadataGroup = h5File.createGroup("metadata");

		// Create attributes in the group. One attribute for each variable in metadata struct
		H5::Attribute basisSizeAttr = metadataGroup.createAttribute("Basis Size", H5::PredType::NATIVE_INT, H5::DataSpace());
		H5::Attribute basisNameAttr = metadataGroup.createAttribute("Basis Type", H5::StrType(H5::PredType::C_S1, metadata.basisName.size()), H5::DataSpace());
		H5::Attribute integratorAttr = metadataGroup.createAttribute("Integrator", H5::StrType(H5::PredType::C_S1, metadata.integrator.size()), H5::DataSpace());

		// Write the attributes
		basisSizeAttr.write(H5::PredType::NATIVE_INT, &metadata.basisSize);
		basisNameAttr.write(H5::StrType(H5::PredType::C_S1, metadata.basisName.size()), metadata.basisName);
		integratorAttr.write(H5::StrType(H5::PredType::C_S1, metadata.integrator.size()), metadata.integrator);

		// Cleanup
		basisSizeAttr.close();
		basisNameAttr.close();
		integratorAttr.close();
		metadataGroup.close();

	} catch (const H5::Exception& error) {
		// Handle HDF5 errors
		std::cerr << "Caught HDF5 exception when writing metadata: " << error.getDetailMsg() << std::endl;
		std::exit(EXIT_FAILURE);
	} catch (const std::exception& error) {
		// Handle other exceptions
		std::cerr << "Caught exception when writing metadata: " << error.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}

}


// Turn Array4D into C-style array, then pass that to overloaded WriteToHDF5()
void writeDataSet(H5::H5File &h5File, const Array4D &data, std::string datasetName) {

	constexpr size_t arrayDimension = 4;
	// 4D array dimensions
	hsize_t dims[arrayDimension] = {data.size(), data[0].size(), data[0][0].size(), data[0][0][0].size()};

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

	// need to pass C-style array
	writeDataSet(h5File, &flattened_data[0], arrayDimension, dims, datasetName);
}

void writeDataSet(H5::H5File &h5File, const double* data, size_t arrayDimension, const hsize_t* dims, std::string datasetName) {

	try {

		// Create dataspace
		H5::DataSpace dataspace(arrayDimension, dims);

		// Create dataset of doubles inside the file/dataspace. Should guarantee correct byte size on any platform
		H5::DataSet dataset = h5File.createDataSet(datasetName, H5::PredType::NATIVE_DOUBLE, dataspace);

		// Write the data
		dataset.write(data, H5::PredType::NATIVE_DOUBLE);

		// Cleanup
		dataset.close();
		dataspace.close();

		std::cout << "Wrote dataset '" << datasetName << "' to " << h5File.getFileName() << std::endl;

	} catch (const H5::Exception& error) {
		// Handle HDF5 errors
		std::cerr << "Caught HDF5 exception: " << error.getDetailMsg() << std::endl;
		std::exit(EXIT_FAILURE);
	} catch (const std::exception& error) {
		// Handle other exceptions
		std::cerr << "Caught exception: " << error.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}

}

void testHDF5() {

	// Produce dummy test data that resembles realistic collision tensor: same size etc  
	const int gridSizeN = 4;
	std::string filename = "testDummy.hdf5";

	Array4D dummyData(gridSizeN - 1, gridSizeN - 1, gridSizeN - 1 , gridSizeN - 1, 0.0);

	// Test RNG
	long seed = 13424213;
	srand48(seed);

	// m,n = Polynomial indices. 
	for (int m = 2; m <= gridSizeN; ++m) for (int n = 1; n <= gridSizeN-1; ++n) {
		// j,k = grid momentum indices 
		for (int j = 1; j <= gridSizeN-1; ++j) for (int k = 1; k <= gridSizeN-1; ++k) {
			dummyData[m-2][n-1][j-1][k-1] = drand48();
		}
	}

	// Create a new HDF5 file. H5F_ACC_TRUNC means we overwrite the file if it exists
	H5::H5File h5File(filename, H5F_ACC_TRUNC);
	H5Metadata metadata;
	metadata.basisSize = gridSizeN;
	metadata.basisName = "Chebyshev";
	metadata.integrator = "Vegas Monte Carlo (GSL)";
	writeMetadata(h5File, metadata);

	writeDataSet(h5File, dummyData, "top");
	writeDataSet(h5File, dummyData, "W");

	h5File.close();
}

