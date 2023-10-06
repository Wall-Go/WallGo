
set(BINARY ${CMAKE_PROJECT_NAME}_tests)

set(TEST_SOURCES test_main.cpp test_integrand.cpp test_integration.cpp)

# Get latest release of Google Test from github 
include(FetchContent)
#Set(FETCHCONTENT_QUIET FALSE) ## Show what the fetching is doing
#set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

## Fetch Google Test library
FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG        v1.13.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

#FetchContent_MakeAvailable(googletest)
## This should be equivalent to FetchContent_MakeAvailable, except that we exclude gtest from install
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
	FetchContent_Populate(googletest)
	add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()



add_executable(${BINARY} ${TEST_SOURCES})

target_include_directories(${BINARY} PRIVATE
	${CMAKE_CURRENT_SOURCE_DIR}/include
)

# link against the 'core' lib and gtest
target_link_libraries(${BINARY} PRIVATE 
	${CMAKE_PROJECT_NAME}_lib
	gtest 
	gtest_main
)

install(TARGETS ${BINARY} DESTINATION ${INSTALL_DIR})

