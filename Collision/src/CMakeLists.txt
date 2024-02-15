
set(BINARY ${CMAKE_PROJECT_NAME})

add_subdirectory(thirdparty)

# source files for core lib (don't include the main function)
set(SOURCES CollisionIntegral.cpp hdf5Interface.cpp gslWrapper.cpp MatrixElement.cpp CollisionManager.cpp)

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(-fPIC)
endif()

add_library(${BINARY}_lib STATIC ${SOURCES})


# main executable. Don't rebuild sources here
add_executable(${BINARY} main.cpp)


## External packages
find_package(HDF5 REQUIRED CXX)
find_package(GSL REQUIRED CXX)

## This is required too but we fetch it if this load fails
find_package(muparser REQUIRED)

if (USE_OMP)
	find_package(OpenMP)
endif()


#---- Link to external libraries
target_link_libraries(${BINARY}_lib PUBLIC
    HDF5::HDF5
    GSL::gsl
    muparser::muparser
    # libraries directly from our thirdparty directory
    tclap
)

if (USE_OMP)
	if (OpenMP_CXX_FOUND)
		target_link_libraries(${BINARY}_lib PUBLIC OpenMP::OpenMP_CXX)
		add_compile_definitions(WITH_OMP=1)
	else()
		message(WARNING "OpenMP installation not found, multithreading will be DISABLED. ")
	endif()
endif()


#---- Header includes
target_include_directories(${BINARY}_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/collision
    ${CMAKE_CURRENT_SOURCE_DIR}/include/thirdparty
    HDF5::HDF5
    GSL::gsl
    muparser::muparser
)

target_link_libraries(${BINARY} PRIVATE ${BINARY}_lib)

install(TARGETS ${BINARY} DESTINATION ${INSTALL_DIR})
