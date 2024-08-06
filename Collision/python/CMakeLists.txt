# What to call the python module. Needs to match the identifier used in PYBIND11_MODULE(...) in C++
set(MODULE PyCollision)

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

pybind11_add_module(${MODULE})

target_sources(${MODULE} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/src/BindToPython.cpp
)

target_link_libraries(${MODULE} PRIVATE
	${WALLGO_LIB}
)	

target_include_directories(${MODULE} PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/src/include"
	${WALLGO_LIB}
)

set(PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/package/WallGoCollision")

add_custom_command(TARGET ${MODULE} POST_BUILD
	COMMENT "Copying module for packaging"
	COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${MODULE}>" "${PACKAGE_DIR}/$<TARGET_FILE_NAME:${MODULE}>"
)

# Generate .pyi stubs. "stubgen" command is from mypy.
# TODO for scikit builds run this always since mypy is a build requirement. For non-scikit, run it only if stubgen command is found
add_custom_command(TARGET ${MODULE} POST_BUILD
	COMMENT "Generating Python stubs for WallGoCollision"
	COMMAND stubgen --include-docstrings -o stubs -m ${MODULE}
	#COMMAND PYTHONPATH="$<TARGET_FILE_DIR:${MODULE}>" ${Python3_EXECUTABLE} -m pybind11_stubgen ${MODULE}
	# move the generated .pyi stub to same directory as our module:
	COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE_DIR:${MODULE}>/stubs/${MODULE}.pyi" "${PACKAGE_DIR}/${MODULE}.pyi"
	WORKING_DIRECTORY "$<TARGET_FILE_DIR:${MODULE}>"
)

# For scikit builds use default install dir so that pip correctly puts the module in site-packages
if(SKBUILD)
	install(TARGETS ${MODULE} DESTINATION .)
else()
	#install(TARGETS ${MODULE} DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/module")
endif()

## Setup a target for test Python script and tell VS how to run it 

set(PYTHON_TEST_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/package/test_collision_package.py")

add_custom_target(
	PybindTestScript ALL
	SOURCES ${PYTHON_TEST_SCRIPT}
	DEPENDS ${MODULE}
	# if we want to run this at build time:
	#COMMAND ${Python3_EXECUTABLE} ${PYTHON_TEST_SCRIPT} "$<TARGET_FILE:${MODULE}>"
)

set_target_properties(PybindTestScript PROPERTIES
	VS_DEBUGGER_COMMAND "${Python3_EXECUTABLE}"
	VS_DEBUGGER_COMMAND_ARGUMENTS "${PYTHON_TEST_SCRIPT}"
	VS_DEBUGGER_ENVIRONMENT "PATH=${Python3_EXECUTABLE_DIR};%PATH%"
#	VS_DEBUGGER_WORKING_DIRECTORY "${PACKAGE_DIR}"
)