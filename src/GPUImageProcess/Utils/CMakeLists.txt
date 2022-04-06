include_directories(${INCLUDE_DIR})

target_sources(${PROJECT_NAME} PUBLIC 
                ${SOURCE_DIR}/GPUImageProcess/Utils/CudaUtils.cpp
                )
