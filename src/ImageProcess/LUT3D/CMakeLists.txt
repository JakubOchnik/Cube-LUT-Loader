include_directories(${INCLUDE_DIR})

target_sources(${PROJECT_NAME} PUBLIC 
                ${SOURCE_DIR}/ImageProcess/LUT3D/applyNearestValue.cpp
                ${SOURCE_DIR}/ImageProcess/LUT3D/applyTrilinear.cpp
                )
