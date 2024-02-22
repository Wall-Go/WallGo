
## Installation

CMake is used as the build system, but dependencies need to be installed first (see below). Once you have installed the dependencies you can compile as:
```
cmake -B build
cmake --build build
cmake --install build
```
This builds and installs a standalone C++ executable to ./bin, example programs to examples/bin and a separate Python module for exposing the C++ code to rest of WallGo.

We compile with OpenMP support by default. Use the ```-DUSE_OMP=Off``` flag to disable OpenMP. You may use ```-DBUILD_PYTHON_MODULE=Off``` to build only a standalone C++ binary without Python bindings.

**Important:** The Python bindings are version dependent and guaranteed to work only with the same version of Python that was used during compilation (we default to the version returned by CMake's FindPython3). If you have multiple Python installations on your system, you can specify the correct version with ```-DUSER_PYTHON_VERSION=3.XX```. If you still have issues, you can try ```-DPython3_ROOT_DIR="path/to/python"``` to specify the location of your preferred Python installation with.


# Installing dependenciens with Conan

Easiest way of handling the dependencies is with the Conan package manager. Requires Conan version > 2.0. The build proceeds as:
```
conan install . --output-folder=build --build=missing
cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
cmake --build build --config Release
cmake --install build
```
**Hint:** Conan can be installed with pip. 


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
