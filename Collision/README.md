
## Installation

CMake is used as the build system, but dependencies need to be installed first (see below). Once you have installed the dependencies you can compile as:
```
cmake -B build [FLAGS]
cmake --build build
cmake --install build
```
This builds and installs a standalone C++ executable to ./bin, example programs to examples/bin and a separate Python module for exposing the C++ code to rest of WallGo.

We compile with OpenMP support by default, needed for parallel evaluation of integrals. Add ```-DUSE_OMP=Off``` in the ```-B``` step to disable OpenMP. You may use ```-DBUILD_PYTHON_MODULE=Off``` to build only a standalone C++ binary without Python bindings.

**Important:** The Python bindings are version dependent and guaranteed to work only with the same version of Python that was used during compilation. We default to the version returned by CMake's FindPython3 which is usually the most recent version of Python. If you have multiple Python installations on your system, you can specify the correct version with ```-DUSER_PYTHON_VERSION=3.XX``` in the ```-B``` step (replace 3.XX with eg. 3.12 for Python 3.12). If you still have issues, you can try ```-DPython3_ROOT_DIR="path/to/python"``` to specify the location of your preferred Python installation.

**Note:** On Windows systems you may have to specify the build configuration explicitly:
```cmake --build build --config Release```

# Installing dependenciens with Conan

Easiest way of handling the dependencies is with the Conan package manager (can be installed with eg. ```pip```). We require Conan version > 2.0. The build proceeds as:
```
conan install . --output-folder=build --build=missing
cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
cmake --build build
cmake --install build
```

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
This installation needs to be global, otherwise pip doesn't install the required CMake files. Alternatively, you could install pybind11 through ```conda``` or build it directly from source.

For Linux systems, ```muparser``` needs to be manually installed from source. Please follow the installation instructions at https://github.com/beltoforion/muparser.