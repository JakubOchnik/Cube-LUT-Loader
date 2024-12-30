include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
                ${SOURCE_DIR}/ImageProcessing/GPU/LUT3DPipelineGPU.cpp
                ${SOURCE_DIR}/ImageProcessing/GPU/GPUModeExecutor.cpp
                ${SOURCE_DIR}/ImageProcessing/GPU/NearestValImplGPU.cpp
                ${SOURCE_DIR}/ImageProcessing/GPU/TrilinearImplGPU.cpp
                )

add_subdirectory(${SOURCE_DIR}/ImageProcessing/GPU/kernels)
add_subdirectory(${SOURCE_DIR}/ImageProcessing/GPU/Utils)
