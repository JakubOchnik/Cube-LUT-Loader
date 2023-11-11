include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
                ${SOURCE_DIR}/ImageProcessing/CPUImageProcess/LUT3D/applyNearestValue.cpp
                ${SOURCE_DIR}/ImageProcessing/CPUImageProcess/LUT3D/applyTrilinear.cpp
                )
