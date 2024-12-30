include_directories(${INCLUDE_DIR})

target_sources(${MAIN_LIB_NAME} PRIVATE
                ${SOURCE_DIR}/ImageProcessing/GPU/kernels/nearestValue.cu
                ${SOURCE_DIR}/ImageProcessing/GPU/kernels/trilinear.cu
                )
