# What to call the python module
set(MODULE WallGoCollisionPy)
# NB: this needs to match the name in PYBIND11_MODULE(name, m) in .cpp

# Installation path for the python module
set(MODULE_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# Find Python. The library produced by pybind11 is version dependent, so we need 
# to allow the user to specify their preferred Python version. 
# Also, Development.Module should be safer than Development. https://pybind11.readthedocs.io/en/stable/compiling.html#findpython-mode

if (DEFINED USER_PYTHON_VERSION)
	find_package(Python3 ${USER_PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter Development.Module)
else()
	find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

find_package(pybind11 CONFIG REQUIRED)

message(STATUS "=== Configured Python bindings for Python ${Python3_VERSION}")

## TODO make static

pybind11_add_module(${MODULE})

target_sources(${MODULE} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/BindToPython.cpp
)

target_link_libraries(${MODULE} PRIVATE
	WallGoCollision
)	

target_include_directories(${MODULE} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/include
	WallGoCollision
)

install(TARGETS ${MODULE} DESTINATION ${MODULE_INSTALL_DIR})
