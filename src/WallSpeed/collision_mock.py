import numpy as np
import h5py # read/write hdf5 structured binary data file format

"""
This is just a quick mock up of the output of the collision integrals, for
setting up the Boltzmann equations.
"""

# hard-coded dimensions of collision array
N_pz = 10 - 1
N_pp = 10 - 1

# hard coded names
file_name = "collision_mock.hdf5"
dataset_name = "top"

# total size and shape of collision array
N_total = (N_pz * N_pp)**2
shape_collision = (N_pz, N_pp, N_pz, N_pp)

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
