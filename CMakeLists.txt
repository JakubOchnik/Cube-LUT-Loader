cmake_minimum_required(VERSION 3.0.0)
project(lut_loader VERSION 0.1.0 LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
add_subdirectory(src)
