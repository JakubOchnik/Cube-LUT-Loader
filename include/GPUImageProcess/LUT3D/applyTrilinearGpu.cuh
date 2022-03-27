#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cmath>
#include <Eigen/Dense>
#include <tuple>

namespace GpuTrilinearDevice {
	void run(dim3 threads, dim3 blocks, unsigned char* image, const char channels, float* LUT, const char LUTsize, const float opacity, const std::tuple<int, int>& imgSize);
}

__global__ void applyTrilinear(unsigned char* image, const char channels, float* LUT, const char LUTsize, const float opacity, int xMax, int yMax);