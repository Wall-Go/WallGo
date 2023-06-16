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
