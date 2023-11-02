# What to call the python module
set(MODULE CollisionModule)
# NB: this needs to match the name in PYBIND11_MODULE(name, m) in .cpp

# Installation path for the python module
set(MODULE_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)


find_package(pybind11 CONFIG REQUIRED)



pybind11_add_module(${MODULE})

target_sources(${MODULE} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/BindToPython.cpp
)

## Link to the main lib. This inherits linkages to its PUBLIC dependencies 

target_link_libraries(${MODULE} PRIVATE
	${CMAKE_PROJECT_NAME}_lib
)	

target_include_directories(${MODULE} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/include
	${CMAKE_PROJECT_NAME}_lib 
)

install(TARGETS ${MODULE} DESTINATION ${MODULE_INSTALL_DIR})
