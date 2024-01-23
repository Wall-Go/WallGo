## Dependencies for the C++ collision module

# Required:

- GSL

- Official HDF5 C++ API (version >= 1.10.1) 

- pybind11 (https://github.com/pybind/pybind11)

- Muparser (https://github.com/beltoforion/muparser/)

# Optional: 

- OpenMP


## Installing dependencies


Linux:
```
sudo apt-get install libgsl-dev libhdf5-dev
```

MacOS: 
```
brew install gsl hdf5 muparser
```

For both systems, pybind11 can be installed using pip:
```
pip install "pybind11[global]"
```
This installation needs to be global, otherwise pip doesn't install the required CMake files. Alternatively you could install pybind11 through conda, or build it directly from source.


For Linux systems, muparser needs to be manually installed from source. Please follow the installation instructions at https://github.com/beltoforion/muparser/. **Note:** muparser can safely be installed without OpenMP support (-DENABLE_OPENMP=OFF) without affecting WallGo/Collision.


## Compiling the Collision module

Stardard CMake build. Go to WallGo/Collision (where the CMakeLists.txt file is) and run:

```
mkdir build
cd build
cmake ..
make
make install	
```

This will produce a standalone C++ executable in build/bin and a separate python module in build/lib. The 'make install' step will copy these to their default locations at ./bin and ./pybind/lib

To only build the C++ program, use the following cmake flag:

```
cmake -DBUILD_PYTHON_MODULE=Off ..
```

If CMake reports errors out due to missing external libraries, please make sure you have installed them as instructed above.

## TODO 

- Figure out how to cancel execution of a long C++ function from Python side. CTRL-C doesn't seem to work; need to kill the process or close the terminal


## Debugging & Profiling

The CMakeLists.txt file defines basic debugging options to use with GCC or Clang compiler. To configure CMake for a debug build, use ```-DCMAKE_BUILD_TYPE=Debug``` when invoking cmake. This will apply compiler flags ```-gp``` and disable optimization flags. Note that the debug version runs much slower than a "Release" build with compiler optimizations enabled.

Running the debug build once should generate file called ```gmon.out``` in the working directory. This can be fed to the popular ```gprof``` profiler by running 

```
gprof <program> gmon.out > analysis.txt
```
where <program> is the executable built with debug flags. The output analysis.txt contains profiling report, eg. information on time spend in each function call.

For visualizing the profiler output you can use the Python package ```gprof2dot```. Note that it requires graphviz to work. Installation on linux:

```
sudo apt install graphviz
pip install gprof2dot
```

Usage example:
```
gprof ./bin/collision | gprof2dot | dot -Tpng -o analysis.png
```
This produces a huge PNG graph that is easier to interpret than the plain text output of gprof.
