include_directories(${INCLUDE_DIR})

target_sources(${PROJECT_NAME} PUBLIC 
                ${SOURCE_DIR}/ImageProcessing/CPUImageProcess/CPUProcessor.cpp
                )

add_subdirectory(${SOURCE_DIR}/ImageProcessing/CPUImageProcess/LUT1D)
add_subdirectory(${SOURCE_DIR}/ImageProcessing/CPUImageProcess/LUT3D)
