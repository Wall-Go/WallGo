cmake_minimum_required(VERSION 3.0)
project(collision VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# Set the path to the installation directories
set(INSTALL_DIR ${CMAKE_SOURCE_DIR}/bin)

# Configure build type if not given by the user
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    #set(CMAKE_BUILD_TYPE Debug)
endif()


set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall")

# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable warnings
    add_compile_options(-Wall -Wextra)
    # Treat all warnings as errors?
    #add_compile_options(-Werror)
    # Enable more warnings (optional)
    #add_compile_options(-Wpedantic)
    
    ## Add flags for gprof profiling. Debug builds only
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
    	add_compile_options(-g)
    	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
		# no optimization here since it can interfere with profiling
    endif()
    
    if(CMAKE_BUILD_TYPE MATCHES "Release")
    	# Optimization
    	add_compile_options(-O3)
    endif()
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Enable warnings and treat them as errors
    #add_compile_options(/W4 /WX)
endif()




## External packages
find_package(HDF5 REQUIRED CXX)
find_package(GSL REQUIRED CXX)

#---- Link to external libraries
link_libraries(
    ${HDF5_LIBRARIES}
    ${GSL_LIBRARIES}
)

#---- And their includes
include_directories(
    ${HDF5_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIRS}
)

## NB: this is linking/including them globally for all targets.
## Not very elegant but makes sense since the same are needed for the python module


#----------------- Target 'collision'. This is the standalone C++ program

set(STANDALONE_EXEC collision)

# Add all source files in the src directory. These will be common also to the python module
file(GLOB COMMON_SOURCES "src/*.cpp")

# Create an executable target
add_executable(${STANDALONE_EXEC} ${COMMON_SOURCES})

# Project specific includes (global scope again) 
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include/collision
)

# Install the standalone program (with 'make install')
install(TARGETS ${STANDALONE_EXEC} DESTINATION ${INSTALL_DIR})

#----------------- End standalone target
 


# Build python module with pybind11 linkage?
option(BUILD_PYTHON_MODULE "Build Python module" ON)
if (BUILD_PYTHON_MODULE)
	add_subdirectory(pybind)
	
endif()


# Enable testing (optional)
#enable_testing()
#add_subdirectory(tests)

