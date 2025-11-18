include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
  ${SOURCE_DIR}/ImageProcessing/OpenCL/OpenCLModeExecutor.cpp
  ${SOURCE_DIR}/ImageProcessing/OpenCL/LUT3DOpenCLImpl.cpp
  ${SOURCE_DIR}/ImageProcessing/OpenCL/LUT1DOpenCLImpl.cpp
                )

add_subdirectory(${SOURCE_DIR}/ImageProcessing/OpenCL/kernels)
add_subdirectory(${SOURCE_DIR}/ImageProcessing/OpenCL/Utils)
