add_executable(${PROJECT_NAME} ${SOURCE_DIR}/main.cpp)

if (BUILD_CUDA)
    find_package(CUDAToolkit)
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")

set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost REQUIRED COMPONENTS program_options)
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

if (BUILD_CUDA)
    message(STATUS "CUDA include dir: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    set(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Boost_INCLUDE_DIRS})
else()
    set(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} Boost::program_options Eigen3::Eigen)

if (BUILD_CUDA)
    add_subdirectory(${SOURCE_DIR}/GPUImageProcess)
endif()
add_subdirectory(${SOURCE_DIR}/ImageProcess)
add_subdirectory(${SOURCE_DIR}/TaskDispatcher)
add_subdirectory(${SOURCE_DIR}/DataLoader)
