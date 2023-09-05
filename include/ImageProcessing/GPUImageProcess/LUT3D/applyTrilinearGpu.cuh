﻿#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <tuple>

namespace GpuTrilinearDevice
{
	void run(dim3 threads, dim3 blocks, unsigned char* image, char channels, float* LUT, int LUTsize, float opacity,
	         const std::tuple<int, int>& imgSize);
}

__global__ void applyTrilinear(unsigned char* image, char channels, const float* LUT, int LUTsize, float opacity,
                               int width, int height);
