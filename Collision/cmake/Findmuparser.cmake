# Findmuparser.cmake

## reference: https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#a-sample-find-module

find_package(PkgConfig)
pkg_check_modules(PC_MUPARSER QUIET muparser)

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

	set(MUPARSER_LIBRARIES ${MUPARSER_LIBRARY})
	set(MUPARSER_INCLUDE_DIRS ${MUPARSER_INCLUDE_DIR})
	set(MUPARSER_DEFINITIONS ${PC_MUPARSER_CFLAGS_OTHER})
	
else()
	## PkgConfig was not able to find the package so it's probably not installed

	## Let's try to fetch it
	include(FetchContent)
	Set(FETCHCONTENT_QUIET FALSE) ## Show what the fetching is doing
	set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

	FetchContent_Declare(
	  muparser
	  GIT_REPOSITORY https://github.com/beltoforion/muparser.git
	  GIT_TAG        v2.3.4
	)
	
	## This is not sufficient - need to manually specify build/config commands...

	FetchContent_GetProperties(muparser)
	if (NOT muparser_POPULATED)
		FetchContent_Populate(muparser)
	endif()

	
endif()
