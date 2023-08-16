include_directories(${INCLUDE_DIR})

target_sources(${PROJECT_NAME} PUBLIC 
                ${SOURCE_DIR}/ImageProcessing/CPUImageProcess/LUT3D/applyNearestValue.cpp
                ${SOURCE_DIR}/ImageProcessing/CPUImageProcess/LUT3D/applyTrilinear.cpp
                )
