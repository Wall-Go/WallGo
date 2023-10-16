cmake_minimum_required(VERSION 3.15)
project(Collision VERSION 0.2.0 LANGUAGES CXX)
# NB: project name is used across other CMake files for target names (${CMAKE_PROJECT_NAME})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

## Installation directory
set(INSTALL_DIR ${CMAKE_SOURCE_DIR}/bin)

# Set default built type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release)
endif()


message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

## Do compile with profiler support? This is off by default
option(PROFILER "Compile with profiler flags" OFF)


# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable warnings
    add_compile_options(-Wall -Wextra)
    # Treat all warnings as errors?
    #add_compile_options(-Werror)
    # Enable more warnings (optional)
    #add_compile_options(-Wpedantic)
    

    
    if (PROFILER)
        # profiler flags needed also for linking
        add_compile_options(-pg)
        add_link_options(-pg)
    	message(STATUS "Profiler flags: -pg")
    endif()


    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		# no optimization here since it can interfere with profiling
		
		option(BUILD_UNIT_TESTS "Build unit tests" ON)
		message(STATUS "==== NOTE: Configuring Debug build")
		message(STATUS "")
		
    endif()
    
    
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    	# Optimization always for non-debug builds
    	add_compile_options(-O3)
    endif()
    
    
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Enable warnings and treat them as errors
    #add_compile_options(/W4 /WX)
endif()



# Will build a 'core' library from code in ./src, apart from main.cpp.
# Using the core lib we then build the 'main' program, a separate unit test module (if in Debug mode) and a separate pybind11 module 

## Main program target
set(MAIN_EXEC collision)

# Code for the core lib + main program
add_subdirectory(src)


# Unit tests
if(BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()


# Pybind11: build the module unless flag was disabled by user 
if(NOT BUILD_PYTHON_MODULE) 
	option(BUILD_PYTHON_MODULE "Build Python module" ON)
endif()


if (BUILD_PYTHON_MODULE)
	add_subdirectory(pybind)
endif()
