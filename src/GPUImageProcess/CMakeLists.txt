include_directories(${INCLUDE_DIR})

target_sources(${PROJECT_NAME} PUBLIC 
                ${SOURCE_DIR}/GPUImageProcess/GPUprocessor.cpp
                )

add_subdirectory(${SOURCE_DIR}/GPUImageProcess/MultidimData)
add_subdirectory(${SOURCE_DIR}/GPUImageProcess/LUT3D)
