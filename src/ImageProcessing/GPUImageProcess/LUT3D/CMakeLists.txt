include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
                ${SOURCE_DIR}/ImageProcessing/GPUImageProcess/LUT3D/applyNearestValueGpu.cu
                ${SOURCE_DIR}/ImageProcessing/GPUImageProcess/LUT3D/applyTrilinearGpu.cu
                ${SOURCE_DIR}/ImageProcessing/GPUImageProcess/LUT3D/applyTrilinearHost.cpp
                ${SOURCE_DIR}/ImageProcessing/GPUImageProcess/LUT3D/applyNearestValueHost.cpp
                )
