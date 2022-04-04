#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <tuple>

#include <GPUImageProcess/Utils/cudaUtils.hpp>

namespace GpuNearestValDevice
{
	void run(dim3 threads, dim3 blocks, unsigned char* image, char channels, float* LUT, char LUTsize, float opacity,
	         const std::tuple<int, int>& imgSize);
}

__global__ void applyNearestKernel(unsigned char* image, char channels, const float* LUT, char LUTsize, float opacity,
                                   int width, int height);
