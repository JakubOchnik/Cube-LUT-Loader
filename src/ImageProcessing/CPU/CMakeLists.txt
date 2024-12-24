include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
                ${SOURCE_DIR}/ImageProcessing/CPU/LUT3DPipelineCPU.cpp
                ${SOURCE_DIR}/ImageProcessing/CPU/TrilinearImplCPU.cpp
                ${SOURCE_DIR}/ImageProcessing/CPU/NearestValImplCPU.cpp
                ${SOURCE_DIR}/ImageProcessing/CPU/Simple1DImplCPU.cpp
                ${SOURCE_DIR}/ImageProcessing/CPU/CPUModeExecutor.cpp
                )
