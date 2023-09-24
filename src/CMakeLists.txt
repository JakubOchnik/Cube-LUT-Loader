add_executable(${PROJECT_NAME} ${SOURCE_DIR}/main.cpp)

# Detect all the required libraries
if (BUILD_CUDA)
    find_package(CUDAToolkit)
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost REQUIRED COMPONENTS program_options)
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# Detect include directories for the external libraries
if (BUILD_CUDA)
    message(STATUS "CUDA include dir: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    set(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Boost_INCLUDE_DIRS})
else()
    set(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
endif()

# Configure the main program library
add_library(${MAIN_LIB_NAME} STATIC)

target_include_directories(${MAIN_LIB_NAME} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${MAIN_LIB_NAME} PUBLIC ${LIBRARIES_TO_LINK})

add_subdirectory(${SOURCE_DIR}/ImageProcessing)
add_subdirectory(${SOURCE_DIR}/TaskDispatcher)
add_subdirectory(${SOURCE_DIR}/DataLoader)

target_link_libraries(${PROJECT_NAME} PRIVATE ${MAIN_LIB_NAME})

# Configure test target
if (BUILD_TESTS)
    set(TEST_DIR ${SOURCE_DIR}/test)
    enable_testing()
    add_subdirectory(test)
endif()
