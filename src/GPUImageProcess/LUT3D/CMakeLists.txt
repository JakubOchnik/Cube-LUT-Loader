include_directories(${INCLUDE_DIR})

target_sources(${PROJECT_NAME} PUBLIC 
                ${SOURCE_DIR}/GPUImageProcess/LUT3D/applyNearestValueGpu.cu
                ${SOURCE_DIR}/GPUImageProcess/LUT3D/applyTrilinearGpu.cu
                ${SOURCE_DIR}/GPUImageProcess/LUT3D/applyTrilinearHost.cpp
                ${SOURCE_DIR}/GPUImageProcess/LUT3D/applyNearestValueHost.cpp
                )
