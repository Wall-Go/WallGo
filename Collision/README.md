========= Dependencies for C++ version ==========

- GSL monte vegas

- pybind11 (TODO)

- Official HDF5 C API headers + library

	To get these on linux:
	sudo apt-get install libhdf5-serial-dev
	sudo apt-get install libhdf5-dev	
	
	
========= Compiling ==========

Stardard CMake build. Go to WallSpeed/Collision (where the CMakeLists.txt file is) and run:

	mkdir build
	cd build
	cmake ..
	make
	
This will produce the executable in build/bin. If cmake errors out due to missing external libraries, install those and try again.
