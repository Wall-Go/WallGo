file(GLOB_RECURSE SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h" "*.inl")

add_library(${WALLGO_LIB} ${SOURCES} ${HEADERS})

set_target_properties(${WALLGO_LIB} PROPERTIES
    POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
)

# Generate export defs (shared lib visibility macros) in the default header dir (easier to #include)
include(GenerateExportHeader)
set(EXPORT_HEADER_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/WallGo/exports_generated.h")
generate_export_header(${WALLGO_LIB}
    EXPORT_FILE_NAME ${EXPORT_HEADER_FILE}
    BASE_NAME WALLGO
)
target_sources(${WALLGO_LIB} PRIVATE ${EXPORT_HEADER_FILE})

## External packages
find_package(HDF5 REQUIRED CXX)
find_package(GSL REQUIRED CXX)
find_package(muparser REQUIRED)

if (USE_OMP)
    find_package(OpenMP)
endif()


#---- Link to external libraries
target_link_libraries(${WALLGO_LIB} PUBLIC
    HDF5::HDF5
    GSL::gsl
    muparser::muparser
)

if (USE_OMP)
	if (OpenMP_CXX_FOUND)
		target_link_libraries(${WALLGO_LIB} PUBLIC OpenMP::OpenMP_CXX)
		add_compile_definitions(WITH_OMP=1)

	else()
		message(WARNING "OpenMP installation not found, multithreading will be DISABLED.")
        add_compile_definitions(WITH_OMP=0)
	endif()
endif()

#---- Header includes

# Public headers: user can #include "WallGo/someheader.h"
target_include_directories(${WALLGO_LIB} PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_include_directories(${WALLGO_LIB} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include/WallGo"
    HDF5::HDF5
    GSL::gsl
    muparser::muparser
)

install(TARGETS ${WALLGO_LIB} DESTINATION ${INSTALL_DIR})