add_executable(${PROJECT_NAME} ${SOURCE_DIR}/main.cpp)

find_package(CUDAToolkit)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")

set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost 1.78 REQUIRED COMPONENTS program_options )
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${INCLUDE_DIR} ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS} CUDA::cudart ${Boost_LIBRARIES})

add_subdirectory(${SOURCE_DIR}/GPUImageProcess)
add_subdirectory(${SOURCE_DIR}/ImageProcess)
add_subdirectory(${SOURCE_DIR}/InitHandler)
add_subdirectory(${SOURCE_DIR}/Loader)
