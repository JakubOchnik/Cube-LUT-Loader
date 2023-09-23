include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PUBLIC
                ${SOURCE_DIR}/ImageProcessing/GPUImageProcess/Utils/CudaUtils.cpp
                )
