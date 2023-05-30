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


# Compiler-specific settings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # Enable warnings
    add_compile_options(-Wall -Wextra)
    # Treat all warnings as errors?
    #add_compile_options(-Werror)
    # Optimization
    add_compile_options(-O3)
    # Enable more warnings (optional)
    #add_compile_options(-Wpedantic)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Enable warnings and treat them as errors
    add_compile_options(/W4 /WX)
endif()

# Configure build type (change to Release for production build)
if(NOT CMAKE_BUILD_TYPE)
    #set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_BUILD_TYPE Debug)
endif()



# Add all source files in the src directory
file(GLOB SOURCES "src/*.cpp")

# Create an executable target
add_executable(collision ${SOURCES})

# Add include directories
target_include_directories(collision
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include/collision
)

## External packages
find_package(HDF5 REQUIRED CXX)
find_package(GSL REQUIRED CXX)

## includes for the packages
target_include_directories(collision 
	PRIVATE ${HDF5_INCLUDE_DIRS}
	PRIVATE ${GSL_INCLUDE_DIRS}
)

# Libraries for the packages
target_link_libraries(collision
	PRIVATE ${HDF5_LIBRARIES}
	PRIVATE ${GSL_LIBRARIES}
)

# Install the standalone program
install(TARGETS collision DESTINATION ${INSTALL_DIR})

## Python module?
add_subdirectory(pybind)


# Add any additional dependencies (if needed)
# find_package(YourDependency REQUIRED)
# target_link_libraries(YourExecutable
#     PRIVATE YourDependency::YourDependency
# )

# Enable testing (optional)
#enable_testing()
#add_subdirectory(tests)

