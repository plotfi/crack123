project(crack)
add_executable(crack crack.cpp)
set(CMAKE_CXX_FLAGS "-std=c++17 -Wc++17-extensions")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
