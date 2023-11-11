include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
                ${SOURCE_DIR}/ImageProcessing/GPUImageProcess/GPUprocessor.cpp
                )

add_subdirectory(${SOURCE_DIR}/ImageProcessing/GPUImageProcess/LUT3D)
add_subdirectory(${SOURCE_DIR}/ImageProcessing/GPUImageProcess/Utils)
