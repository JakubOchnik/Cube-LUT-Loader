#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cmath>
#include <tuple>

namespace GpuNearestValDevice 
{
	void run(dim3 threads, dim3 blocks, unsigned char* image, const char channels, float* LUT, const char LUTsize, const float opacity, const std::tuple<int, int>& imgSize);
}

__global__ void applyNearestKernel(unsigned char* image, char channels, float* LUT, char LUTsize, float opacity, int xMax, int yMax);

