# What to call the python module. Needs to match the identifier used in PYBIND11_MODULE(...) in C++
set(MODULENAME _WallGoCollision)

# Find Python. The library produced by pybind11 is version dependent,
# so we must to allow the user to specify their preferred Python version. 
# Also, Development.Module should be safer than Development. https://pybind11.readthedocs.io/en/stable/compiling.html#findpython-mode
if (DEFINED USER_PYTHON_VERSION)
	find_package(Python3 ${USER_PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter Development.Module)
else()
	find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

## NOTE: Probable issue with Debug builds on Windows, see https://github.com/pybind/pybind11/issues/3403.
## TLDR: find_python3 gives python_d.lib but pybind11 links against the non-debug lib.
## Here's a hack that makes us always use the Release version of python lib
set_target_properties(Python3::Module PROPERTIES
        MAP_IMPORTED_CONFIG_DEBUG ";RELEASE"
)

find_package(pybind11 CONFIG REQUIRED)

message(STATUS "=== Building Python bindings for Python version ${Python3_VERSION}")

pybind11_add_module(${MODULENAME})

target_compile_definitions(${MODULENAME} PUBLIC WG_PYTHON_MODULE_NAME=${MODULENAME})

target_sources(${MODULENAME} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/BindToPython.cpp
)

target_link_libraries(${MODULENAME} PRIVATE
	${WALLGO_LIB}
)	

target_include_directories(${MODULENAME} PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/src/include"
	${WALLGO_LIB}
)

# Specify where our __init__.py and other python-package-specific files are. Stubs will be produced in this folder
set(PACKAGE_SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/packagedata/WallGoCollision")

# Generate .pyi stubs. "stubgen" command is from mypy.
# This step is non-fatal, so we continue with the build even if the stubgen fails (achieved with logical || in the command).
add_custom_command(TARGET ${MODULENAME} POST_BUILD
	COMMENT "Generating Python stubs for WallGoCollision"
	COMMAND stubgen --include-docstrings -o stubs -m ${MODULENAME} || (exit 0)
	# Old version that uses pybind11_stubgen instead:
	#COMMAND PYTHONPATH="$<TARGET_FILE_DIR:${MODULENAME}>" ${Python3_EXECUTABLE} -m pybind11_stubgen ${MODULENAME}

	# move the generated .pyi stub to same directory as our module:
	COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE_DIR:${MODULENAME}>/stubs/${MODULENAME}.pyi" "${PACKAGE_SOURCE_DIRECTORY}/${MODULENAME}.pyi" || (exit 0)
	WORKING_DIRECTORY "$<TARGET_FILE_DIR:${MODULENAME}>"
)

if (SKBUILD)
	# Scikit handles installation to correct location
	install(TARGETS ${MODULENAME} DESTINATION ${SKBUILD_PROJECT_NAME})
else()

	set(PACKAGE_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/package_standalone/WallGoCollision")

	add_custom_command(TARGET ${MODULENAME} POST_BUILD
		COMMENT "Copy Python package data to 'package_standalone/'"
		COMMAND ${CMAKE_COMMAND} -E copy_directory "${PACKAGE_SOURCE_DIRECTORY}" "${PACKAGE_INSTALL_DIR}" 
	)

	install(TARGETS ${MODULENAME} DESTINATION "${PACKAGE_INSTALL_DIR}")
endif()

if(!SKBUILD)
	## Setup a target for test Python script and tell VS how to run it 
	set(PYTHON_TEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/package_standalone/test_collision_package.py")

	add_custom_target(
		PybindTestScript ALL
		SOURCES ${PYTHON_TEST_SCRIPT}
		DEPENDS ${MODULENAME}
	)

	set_target_properties(PybindTestScript PROPERTIES
		VS_DEBUGGER_COMMAND "${Python3_EXECUTABLE}"
		VS_DEBUGGER_COMMAND_ARGUMENTS "${PYTHON_TEST_SCRIPT}"
		VS_DEBUGGER_ENVIRONMENT "PATH=${Python3_EXECUTABLE_DIR};%PATH%"
	#	VS_DEBUGGER_WORKING_DIRECTORY "${PACKAGE_DIR}"
	)
endif()