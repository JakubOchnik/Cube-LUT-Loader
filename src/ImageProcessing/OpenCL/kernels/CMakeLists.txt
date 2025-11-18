include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
  ${SOURCE_DIR}/ImageProcessing/OpenCL/kernels/nearestVelue.cl
  ${SOURCE_DIR}/ImageProcessing/OpenCL/kernels/linear3DValue.cl
  ${SOURCE_DIR}/ImageProcessing/OpenCL/kernels/linear1DValue.cl
                )
