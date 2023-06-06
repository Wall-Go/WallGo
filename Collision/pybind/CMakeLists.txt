cmake_minimum_required(VERSION 3.00)

# What to call the python module
set(MODULE_NAME "CollisionModule")

project(${MODULE_NAME} LANGUAGES CXX)


# Installation path for the python module
set(MODULE_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

# Find pybind11 package
find_package(pybind11 CONFIG REQUIRED)

# Create the pybind11 module (this takes place of add_executable
pybind11_add_module(${MODULE_NAME})

# If this file gets called from CMakeLists.txt elsewhere then 
# CMAKE_SOURCE_DIR = location of the original calling file
# CMAKE_CURRENT_SOURCE_DIR = location of this file as it gets processed


target_include_directories(${MODULE_NAME}
	PRIVATE ${CMAKE_SOURCE_DIR}/pybind/src/include
)


target_sources(${MODULE_NAME} 
	PRIVATE ${COMMON_SOURCES}	# from the standalone executable
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/BindToPython.cpp
)


# Install the python module ('make install')
install(TARGETS ${MODULE_NAME} DESTINATION ${MODULE_INSTALL_DIR})
