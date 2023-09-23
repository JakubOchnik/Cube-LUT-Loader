cmake_minimum_required(VERSION 3.0.0)

# Toggle building GPU accelerated version
set(BUILD_CUDA OFF)

if(BUILD_CUDA)
    project(lut_loader VERSION 0.1.0 LANGUAGES CUDA CXX)
    add_compile_definitions(BUILD_CUDA)
else()
    project(lut_loader VERSION 0.1.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)

option(ENABLE_CODE_ANALYSIS "Run code analysis" OFF)
if(ENABLE_CODE_ANALYSIS)
    set(CMAKE_CXX_CPPCHECK "cppcheck --enable=all --std=c++17")
endif()

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
add_subdirectory(src)
