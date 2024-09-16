#include <iostream>

#include <cmath>
#include <vector>

#include "hdf5Interface.h"
#include "ResultContainers.h"

namespace wallgo
{

namespace utils
{

void writeMetadata(H5::H5File& h5File, const CollisionMetadata& metadata)
{
	try
	{
		// Create a group to hold metadata (keeping it separate from the actual data)
		H5::Group metadataGroup = h5File.createGroup("Metadata");

		// Create attributes in the group. One attribute for each variable in metadata struct
		H5::Attribute basisSizeAttr = metadataGroup.createAttribute("Basis Size", H5::PredType::NATIVE_INT, H5::DataSpace());
		H5::Attribute basisNameAttr = metadataGroup.createAttribute("Basis Type", H5::StrType(H5::PredType::C_S1, metadata.basisName.size()), H5::DataSpace());
		H5::Attribute integratorAttr = metadataGroup.createAttribute("Integrator", H5::StrType(H5::PredType::C_S1, metadata.integrator.size()), H5::DataSpace());
		H5::Attribute seedAttr = metadataGroup.createAttribute("Seed", H5::PredType::NATIVE_INT, H5::DataSpace());
		H5::Attribute numThreadsAttr = metadataGroup.createAttribute("Num Threads", H5::PredType::NATIVE_INT, H5::DataSpace());
		H5::Attribute timeSpentAttr = metadataGroup.createAttribute("Evaluation Time in Seconds", H5::PredType::NATIVE_INT, H5::DataSpace());
		// Integration options
		H5::Attribute callsAttr = metadataGroup.createAttribute("Calls Per Vegas Step", H5::PredType::NATIVE_INT, H5::DataSpace());
		H5::Attribute maxTriesAttr = metadataGroup.createAttribute("Vegas Max Tries", H5::PredType::NATIVE_INT, H5::DataSpace());
		H5::Attribute absoluteErrorGoalAttr = metadataGroup.createAttribute("Absolute Tolerance", H5::PredType::NATIVE_DOUBLE, H5::DataSpace());
		H5::Attribute relativeErrorGoalAttr = metadataGroup.createAttribute("Relative Tolerance", H5::PredType::NATIVE_DOUBLE, H5::DataSpace());
		H5::Attribute maxIntegrationMomentumAttr = metadataGroup.createAttribute("Max Integration Momentum", H5::PredType::NATIVE_DOUBLE, H5::DataSpace());
		H5::Attribute optimizeUltrarelativisticAttr = metadataGroup.createAttribute("bOptimizeUltrarelativistic", H5::PredType::NATIVE_INT, H5::DataSpace());
		
		uint32_t optimizeUltrarelativistic = metadata.usedIntegrationOptions.bOptimizeUltrarelativistic ? 1 : 0;

		// Write the attributes
		basisSizeAttr.write(H5::PredType::NATIVE_INT, &metadata.basisSize);
		basisNameAttr.write(H5::StrType(H5::PredType::C_S1, metadata.basisName.size()), metadata.basisName);
		integratorAttr.write(H5::StrType(H5::PredType::C_S1, metadata.integrator.size()), metadata.integrator);
		seedAttr.write(H5::PredType::NATIVE_INT, &metadata.seed);
		numThreadsAttr.write(H5::PredType::NATIVE_INT, &metadata.numThreads);
		timeSpentAttr.write(H5::PredType::NATIVE_INT, &metadata.timeSpent);
		//
		callsAttr.write(H5::PredType::NATIVE_INT, &metadata.usedIntegrationOptions.calls);
		maxTriesAttr.write(H5::PredType::NATIVE_INT, &metadata.usedIntegrationOptions.maxTries);
		absoluteErrorGoalAttr.write(H5::PredType::NATIVE_DOUBLE, &metadata.usedIntegrationOptions.absoluteErrorGoal);
		relativeErrorGoalAttr.write(H5::PredType::NATIVE_DOUBLE, &metadata.usedIntegrationOptions.relativeErrorGoal);
		maxIntegrationMomentumAttr.write(H5::PredType::NATIVE_DOUBLE, &metadata.usedIntegrationOptions.maxIntegrationMomentum);
		optimizeUltrarelativisticAttr.write(H5::PredType::NATIVE_INT, &optimizeUltrarelativistic);

		metadataGroup.close();

		// Write model params in a separate group
		H5::Group modelParamsGroup = h5File.createGroup("Model Parameters");
		for (const auto& [key, value] : metadata.modelParameters.getParameterMap())
		{
			H5::Attribute paramAttr = modelParamsGroup.createAttribute(key, H5::PredType::NATIVE_DOUBLE, H5::DataSpace());
			paramAttr.write(H5::PredType::NATIVE_DOUBLE, &value);
		}

		modelParamsGroup.close();

	}
	catch (const H5::Exception& error)
	{
		// Handle HDF5 errors
		std::cerr << "Caught HDF5 exception when writing metadata: " << error.getDetailMsg() << std::endl;
		std::exit(EXIT_FAILURE);
	}
	catch (const std::exception& error)
	{
		// Handle other exceptions
		std::cerr << "Caught exception when writing metadata: " << error.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

// Turn Array4D into C-style array, then pass that to overloaded WriteToHDF5()
void writeDataSet(H5::H5File& h5File, const Array4D& data, std::string datasetName) {

	constexpr size_t arrayDimension = 4;
	// 4D array dimensions
	hsize_t dims[arrayDimension] = { data.size(), data[0].size(), data[0][0].size(), data[0][0][0].size() };

	// For writing we need to pass the data as a contiguous block of memory. A nested std::vector is not
	// guaranteed to be contiguous so need to flatten the data here
	std::vector<double> flattened_data;
	for (const auto& vec3 : data)
	{
		for (const auto& vec2 : vec3)
		{
			for (const auto& vec1 : vec2)
			{
				flattened_data.insert(flattened_data.end(), vec1.begin(), vec1.end());
			}
		}
	}

	// need to pass C-style array
	writeDataSet(h5File, &flattened_data[0], arrayDimension, dims, datasetName);
}

void writeDataSet(H5::H5File& h5File, const double* data, size_t arrayDimension, const hsize_t* dims, std::string datasetName)
{
	try
	{

		// Create dataspace
		H5::DataSpace dataspace(static_cast<int>(arrayDimension), dims);

		// Create dataset of doubles inside the file/dataspace. Should guarantee correct byte size on any platform
		H5::DataSet dataset = h5File.createDataSet(datasetName, H5::PredType::NATIVE_DOUBLE, dataspace);

		// Write the data
		dataset.write(data, H5::PredType::NATIVE_DOUBLE);

		// Cleanup
		dataset.close();
		dataspace.close();

		std::cout << "Wrote dataset '" << datasetName << "' to " << h5File.getFileName() << std::endl;

	}
	catch (const H5::Exception& error)
	{
		// Handle HDF5 errors
		std::cerr << "Caught HDF5 exception: " << error.getDetailMsg() << std::endl;
		std::exit(EXIT_FAILURE);
	}
	catch (const std::exception& error)
	{
		// Handle other exceptions
		std::cerr << "Caught exception: " << error.what() << std::endl;
		std::exit(EXIT_FAILURE);
	}

}

} // namespace utils

} // namespace wallgo