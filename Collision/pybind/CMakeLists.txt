# What to call the python module
set(MODULE WallGoCollisionPy)
# NB: this needs to match the name in PYBIND11_MODULE(name, m) in .cpp

# Installation path for the python module
set(MODULE_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Explicitly require Python package, otherwise pybind11 throwed errors on some platforms
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(${MODULE})

target_sources(${MODULE} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/BindToPython.cpp
)

## Link to the main lib. This inherits linkages to its PUBLIC dependencies

target_link_libraries(${MODULE} PRIVATE
	WallGoCollision
)	

target_include_directories(${MODULE} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/include
	WallGoCollision
)

install(TARGETS ${MODULE} DESTINATION ${MODULE_INSTALL_DIR})
