
## CMakeLists.txt for building WallGo/Collision C++ examples

add_executable(QCD QCD.cpp)

target_link_libraries(QCD PRIVATE WallGoCollision)

install(TARGETS QCD DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/bin)
