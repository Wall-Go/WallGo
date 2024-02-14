## Dependencies for the C++ collision module

# Required:

- GSL

- Official HDF5 C++ API (version >= 1.10.1) 

- pybind11 (https://github.com/pybind/pybind11)

- Muparser (https://github.com/beltoforion/muparser/)

# Optional: 

- OpenMP


## Installation

CMake is used as the build system, but dependencies need to be installed first. 

# Using Conan

Easiest way of handling the dependencies is with Conan (version > 2.0):
```
conan install . --output-folder=build --build=missing
cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
cmake --build build
cmake --install build
```
This will build and install all dependencies, a standalone C++ executable and a separate Python module that exposes the C++ code to rest of WallGo. If you're compiling on Windows with the MSVC compiler, add ```--config Release``` in the ```cmake --build``` step.


# Manually installing dependencies

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

For Linux systems, muparser needs to be manually installed from source. Please follow the installation instructions at https://github.com/beltoforion/muparser/. **Note:** muparser can safely be installed without OpenMP support (```-DENABLE_OPENMP=OFF```) without affecting WallGo/Collision.


Then proceed with standard CMake build:
```
cmake -B build
cmake --build build
cmake --install build
```

# Optional build options

Following options are available in the CMake configure step:
- ```-DBUILD_PYTHON_MODULE=Off```: Build only a standalone binary without Python bindings. pybind11 is not required with this option.
- ```-DUSE_OMP=Off```: Disables OpenMP support.


## Debugging & Profiling [for developers!]

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
