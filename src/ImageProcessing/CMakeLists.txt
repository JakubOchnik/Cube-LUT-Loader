include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
                ${SOURCE_DIR}/ImageProcessing/ImageProcessExecutor.cpp
                )

add_subdirectory(${SOURCE_DIR}/ImageProcessing/CPU)
if (BUILD_CUDA)
    add_subdirectory(${SOURCE_DIR}/ImageProcessing/GPUImageProcess)
endif()
