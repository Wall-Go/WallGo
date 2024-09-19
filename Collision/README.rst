
## Installation

You must have a C++17 compliant compiler (gcc, clang, MSVC etc) and CMake (version 3.18 or newer) installed.

# Installing the Python extension module only (with pip)

```pip install . -v```

This works by invoking a CMake build via the scikit-core-build pyproject backend, and its Conan extension for handling dependencies with other C++ libraries.
It produces a Python package that gets installed to pip's default location (usually ```site-packages/```). To use in your Python projects you only have to ```import WallGoCollision```.

A Python stub file (provides type hints and docstrings) is also generated and installed with the package.

Note that the pip installer is still work in progress. The following limitations apply:
- OpenMP is not included as an explicit dependency, so it does not automatically get installed.
The resulting module will still be multithreaded if you have an available OpenMP installation.
TODO: Conan center has llvm-openmp available so we could automatically fetch it, but the issue about bad profile detection needs to be solved first (see below)
because the library fails to compile with default profiles on older compilers (eg. gcc 9).

- The automatic generation of stubs may fail (happened when building on a virtual machine). This is non fatal and the main module should still install normally. If you have ```mypy``` installed you can generate the stubs manually:
```stubgen --include_docstrings -o stubs -m WallGoCollision._WallGoCollision```
Copy the .pyi files from the generated stubs/ folder to where the WallGoCollision package was installed.

- The installation uses Conan to download and compile C++ libraries that WallGoCollisions depends on.
Currently this always uses the ```default``` Conan profile and creates the profile if it's not found (eg. if you don't already have Conan installed).
The automated profile creation is done with the ```conan profile detect``` command which tries to guess a good compiler configuration based on what is available on your system.
The profile detection is not perfect, so if you find that Conan is failing to compile dependencies (eg. cppstd version too low),
you can install Conan yourself (```pip install Conan```) and modify your default profile as necessary.
For example, the default C++ standard (```cppstd```) in gcc 9 is gnu14 and may not be sufficient for compiling all dependencies, even though the compiler fully supports C++17.

TODO: Compile binaries using Github workflows for common OS/Python version combinations and upload those to PyPi. 

# CMake installation

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

Easiest way of handling the dependencies is with the Conan package manager (can be installed with eg. ```pip```). We require Conan version >= 2.0. The build proceeds as:
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
