#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cmath>

namespace TrilinearGpu {
	void run(dim3 grid, unsigned char* image, char channels, float* LUT, char LUTsize, float opacity);
}

__global__ void applyTrilinear(unsigned char* image, char channels, float* LUT, char LUTsize, float opacity);