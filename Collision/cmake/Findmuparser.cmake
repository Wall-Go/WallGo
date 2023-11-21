# Findmuparser.cmake

## reference: https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#a-sample-find-module

find_package(PkgConfig)
pkg_check_modules(PC_MUPARSER muparser)


find_path(MUPARSER_INCLUDE_DIR
	NAMES muParser.h
	PATHS ${PC_MUPARSER_INCLUDE_DIRS}
	PATH_SUFFIXES MUPARSER
)

find_library(MUPARSER_LIBRARY
	NAMES muparser
	PATHS ${PC_MUPARSER_LIBRARY_DIRS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(muparser
    FOUND_VAR MUPARSER_FOUND
    REQUIRED_VARS
        MUPARSER_LIBRARY
        MUPARSER_INCLUDE_DIR
    VERSION_VAR MUPARSER_VERSION
)


if(MUPARSER_FOUND)

    if (NOT TARGET muparser::muparser)
        ## define imported target
        add_library(muparser::muparser UNKNOWN IMPORTED)
        
        set_target_properties(muparser::muparser
            PROPERTIES IMPORTED_LOCATION ${MUPARSER_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${MUPARSER_INCLUDE_DIR}
        )

    endif()

	set(MUPARSER_LIBRARIES ${MUPARSER_LIBRARY})
	set(MUPARSER_INCLUDE_DIRS ${MUPARSER_INCLUDE_DIR})
	set(MUPARSER_DEFINITIONS ${PC_MUPARSER_CFLAGS_OTHER})

endif()
