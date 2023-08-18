import numpy as np
import h5py # read/write hdf5 structured binary data file format

"""
This is just a quick mock up of the output of the collision integrals, for
setting up the Boltzmann equations.
"""

fileName = "collisions_N20.hdf5"

try:
    # Open the HDF5 file in read-only mode
    with h5py.File(fileName, "r") as file:

        # Read and print the metadata. This is a HDF5 group of attributes
        print("\n=== Metadata ===")
        metadata = file["metadata"]
        ## The hardcoded names here need match attribute labels in the file
        basisSize = metadata.attrs["Basis Size"]
        basisType = metadata.attrs["Basis Type"]
        integrator = metadata.attrs["Integrator"]
        print([type(basisSize), type(basisType), type(integrator)])
        print("Basis Size: ", basisSize) # this is N
        print("Basis Type: ", basisType) # this just tells what polynomial basis we are using. Just Chebyshev for now
        print("Integrator: ", integrator) # what numerical integrator was used


        # Read and print the dataset names. Need to manually drop metadata from here.
        datasetNames = list(file.keys())
        datasetNames.remove("metadata")
        print("\n=== Dataset names ===")
        for name in datasetNames:
            print(name)

        # Read and print the data arrays. This would crash if metadata was included here: it is a group and not dataset
        print("\n=== Data arrays ===")
        for name in datasetNames:
            dataset = file[name]
            data = dataset[:]
            print("\n-- %s --" % name)
            #print(data)

        ## To pick eg. the collision tensor for top quark:
        dataset = file["top"]
        collisionsTop = dataset[:]
        print("\n=== EXAMPLE: top quark collisions ===")

        ## Note that the array indices run from 0 to N-1 while the "grid indices" used in the paper are m = 2, ... N and n,j,k = 1, ... N-1
        ## so need to offset accordingly
        m, n, j, k = 2, 1, 1, 1
        print("C[%d, %d, %d, %d] = %g" % (m, n, j, k, collisionsTop[m-2, n-1, j-1, k-1]) )



except Exception as error:
    # Handle any errors that occur
    print("Error:", str(error))

'''
# hard-coded dimensions of collision array
N_pz = 20
N_pp = 20

# hard coded names
file_name = "collision_mock.hdf5"
dataset_name = "top"

# total size and shape of collision array
N_total = ((N_pz - 1) * (N_pp - 1))**2
shape_collision = (N_pz - 1, N_pp - 1, N_pz - 1, N_pp - 1)

# costructing mock data, just random numbers for now
collision = np.random.rand(*shape_collision)

# printing the first point
print("collision[0, 0, 0, 0] =", collision[0, 0, 0, 0])

# writing mock data to file
with h5py.File(file_name, "w") as file:
    data_set = file.create_dataset(dataset_name, data=collision)

# reading the data back from file, to test that it worked
with h5py.File(file_name, "r") as file:
    collision_read = np.array(file[dataset_name])

# printing the first point again
print("collision_read[0, 0, 0, 0] =", collision_read[0, 0, 0, 0])
'''
