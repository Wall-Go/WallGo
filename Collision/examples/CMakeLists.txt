
## CMakeLists.txt for building WallGo/Collision C++ examples

add_executable(QCD QCD.cpp)

target_link_libraries(QCD PRIVATE WallGoCollision)

install(TARGETS QCD DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Set working directory for VS debugger
set_property(TARGET QCD PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# And VS startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT QCD)