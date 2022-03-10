include_directories(${INCLUDE_DIR})

target_sources(${PROJECT_NAME} PUBLIC 
                ${SOURCE_DIR}/ImageProcess/Processor.cpp
                )

add_subdirectory(${SOURCE_DIR}/ImageProcess/LUT1D)
add_subdirectory(${SOURCE_DIR}/ImageProcess/LUT3D)
