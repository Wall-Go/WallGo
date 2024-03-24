cmake_minimum_required(VERSION 3.18)
project(Collision VERSION 0.2.0 LANGUAGES CXX)
# NB: project name is used across other CMake files for target names (${CMAKE_PROJECT_NAME})

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)


set(CMAKE_CXX_STANDARD 17)
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


MESSAGE(STATUS "Using toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

## Do compile with profiler support? This is off by default
option(PROFILER "Compile with profiler flags" OFF)


## Use OpenMP by default
option(USE_OMP "Build with OpenMP support" ON)
option(BUILD_EXAMPLES "Build WallGo/Collision C++ examples" ON)

## For MacOS, may need to manually setup OpenMP
if(APPLE AND USE_OMP)

    ## not good! TODO
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_C_LIB_NAMES "libomp")
        set(OpenMP_omp_LIBRARY omp)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "libomp")
        set(OpenMP_omp_LIBRARY omp)
    endif()
endif()

# Build the Python module unless the user disabled this
option(BUILD_PYTHON_MODULE "Build Python module" ON)

if(BUILD_PYTHON_MODULE)
	add_compile_definitions(WITH_PYTHON=1) ## C++: #if WITH_PYTHON
else()
    add_compile_definitions(WITH_PYTHON=0)
endif()


# Compiler-specific settings. We don't currently support debug flags etc for MSVC
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable warnings
    add_compile_options(-Wall -Wextra)

    #add_compile_options(-fvisibility=hidden)

    if (PROFILER)
        # profiler flags needed also for linking
        add_compile_options(-pg)
        add_link_options(-pg)
    	message(STATUS "Profiler flags: -pg")
    endif()


    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
		option(BUILD_UNIT_TESTS "Build unit tests" ON)
		message(STATUS "==== NOTE: Configuring Debug build")
		message(STATUS "")

        # Treat all warnings as errors?
        #add_compile_options(-Werror)
        # Allow compiler to become annoying
        add_compile_options(-Wpedantic)
    endif()

    if (NOT USE_OMP)
        # Disable pragma warnings, otherwise each #pragma omp would throw a warning
        add_compile_options(-Wno-unknown-pragmas)
    endif()

endif()

## Release build specific build options
if (CMAKE_BUILD_TYPE STREQUAL "Release")

    add_compile_definitions(NDEBUG)

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        add_compile_options(-O3)
    endif()

endif()


add_subdirectory(src)


if (BUILD_PYTHON_MODULE)
	add_subdirectory(pybind)
endif()

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_UNIT_TESTS)
    add_subdirectory(tests)
endif()
