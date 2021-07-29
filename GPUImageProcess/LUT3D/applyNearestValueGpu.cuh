#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cmath>

namespace NearestValGpu {
	void run(dim3 grid, unsigned char* image, char channels, float* LUT, char LUTsize, float opacity);
}

__device__ float* mul(float* a, int offset, const float val);

__device__ float* sum(float* a, float* b);

__device__ int l_index(int r, int g, int b, char LUTsize, char channels);

__global__ void applyNearestKernel(unsigned char* image, char channels, float* LUT, char LUTsize, float opacity);

