cmake_minimum_required(VERSION 3.5.0)

option(BUILD_TESTS "Build tests" OFF)
option(BUILD_CUDA "Build project with CUDA libraries" OFF)

if(BUILD_CUDA)
    project(lut_loader VERSION 0.1.0 LANGUAGES CUDA CXX)
    add_compile_definitions(BUILD_CUDA)
else()
    project(lut_loader VERSION 0.1.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(MAIN_LIB_NAME lut_loader_base)
set(LIBRARIES_TO_LINK opencv_core opencv_imgcodecs Boost::program_options Eigen3::Eigen)
add_subdirectory(src)
