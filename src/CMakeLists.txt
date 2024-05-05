# Detect all the required libraries
if (BUILD_CUDA)
    find_package(CUDAToolkit)
endif()

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")

find_package(Boost REQUIRED COMPONENTS program_options)
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)

# Configure the main program library
add_library(${MAIN_LIB_NAME} STATIC)

# Detect include directories for the external libraries
if (BUILD_CUDA)
    message(STATUS "CUDA include dir: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    set(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${INCLUDE_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${Boost_INCLUDE_DIRS})

    # Silence irrelevant Eigen3 warnings
    if (MSVC)
        set(MSVC_UNKNOWN_PRAGMA "-Xcompiler" "/wd4068")
    endif()

    target_compile_options(${MAIN_LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --diag-suppress 20012,20014 
        --expt-relaxed-constexpr 
        ${MSVC_UNKNOWN_PRAGMA}>
    )

else()
    set(INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS} ${INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
endif()

target_include_directories(${MAIN_LIB_NAME} PUBLIC ${INCLUDE_DIRS})
target_link_libraries(${MAIN_LIB_NAME} PUBLIC ${LIBRARIES_TO_LINK})

add_subdirectory(${SOURCE_DIR}/ImageProcessing)
add_subdirectory(${SOURCE_DIR}/TaskDispatcher)
add_subdirectory(${SOURCE_DIR}/FileIO)

add_executable(${PROJECT_NAME} ${SOURCE_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${MAIN_LIB_NAME})

# Configure test target
if (BUILD_TESTS)
    set(TEST_DIR ${SOURCE_DIR}/test)
    add_subdirectory(test)
endif()
