#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Eigen/Dense>
#include <tuple>

#include <GPUImageProcess/Utils/CudaUtils.hpp>

namespace GpuTrilinearDevice
{
	void run(dim3 threads, dim3 blocks, unsigned char* image, char channels, float* LUT, int LUTsize, float opacity,
	         const std::tuple<int, int>& imgSize);
}

__global__ void applyTrilinear(unsigned char* image, char channels, const float* LUT, int LUTsize, float opacity,
                               int xMax, int yMax);
