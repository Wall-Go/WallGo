
set(BINARY ${CMAKE_PROJECT_NAME})

# source files for core lib (don't include the main function)
set(SOURCES CollisionIntegral.cpp WriteToHDF5.cpp)

add_library(${BINARY}_lib STATIC ${SOURCES})

# main executable. Don't rebuild sources here
add_executable(${BINARY} main.cpp)

## External packages
find_package(HDF5 REQUIRED CXX)
find_package(GSL REQUIRED CXX)
find_package(OpenMP)



#---- Link to external libraries
target_link_libraries(${BINARY}_lib PUBLIC
    ${HDF5_LIBRARIES}
    ${GSL_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(${BINARY}_lib PUBLIC OpenMP::OpenMP_CXX)
else()
    message(STATUS "==== NOTE: OpenMP installation not found, disabling multithreading")
endif()

#---- Header includes
target_include_directories(${BINARY}_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/collision
    ${CMAKE_CURRENT_SOURCE_DIR}/include/thirdparty
    ${HDF5_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIRS}
)

target_link_libraries(${BINARY} PRIVATE ${BINARY}_lib)

# 'make install'
install(TARGETS ${BINARY} DESTINATION ${INSTALL_DIR})