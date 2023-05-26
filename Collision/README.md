========= Dependencies for C++ version ==========

- GSL monte vegas

- pybind11 (TODO)

- Official HDF5 C API library



----- Installing dependencies -----

Linux:
	sudo apt-get install libgsl-dev
	sudo apt-get install libhdf5-dev

MacOS: 
	brew install gsl
	brew install hdf5

		
========= Compiling ==========

Stardard CMake build. Go to WallSpeed/Collision (where the CMakeLists.txt file is) and run:

	mkdir build
	cd build
	cmake ..
	make
	
This will produce the executable in build/bin. If cmake errors out due to missing external libraries, install those and try again.
