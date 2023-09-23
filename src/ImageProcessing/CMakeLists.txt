include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PUBLIC
                ${SOURCE_DIR}/ImageProcessing/ImageProcessor.cpp
                )

add_subdirectory(${SOURCE_DIR}/ImageProcessing/CPUImageProcess)
if (BUILD_CUDA)
    add_subdirectory(${SOURCE_DIR}/ImageProcessing/GPUImageProcess)
endif()
