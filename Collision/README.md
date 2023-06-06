## Dependencies for the C++ collision module

- GSL monte vegas

- Official HDF5 C API library

- pybind11



## Installing dependencies


Linux:
```
	sudo apt-get install libgsl-dev
	sudo apt-get install libhdf5-dev
```

MacOS: 
```
	brew install gsl
	brew install hdf5
```

For both systems the python bindings can be installed with pip:
```
	pip install pybind11[global]
```

The pybind11 installation needs to be global, otherwise pip doesn't install the required CMake files. Alternatively you could install pybind11 through conda (have not tested).


## Compiling

Stardard CMake build. Go to WallSpeed/Collision (where the CMakeLists.txt file is) and run:

```
mkdir build
cd build
cmake ..
make
make install	
```

This will produce a standalone C++ executable in build/bin and a separate python module in build/lib. The 'make install' step will copy these to their defaultl locations at ./bin and ./pybind/lib

To only build the C++ program, use the following cmake flag:

```
cmake -DBUILD_PYTHON_MODULE=Off ..
```

If cmake errors out due to missing external libraries, install those and try again.

## TODO 

- Check that the program builds fine without pybind11 installed if -DBUILD_PYTHON_MODULE=Off is used
