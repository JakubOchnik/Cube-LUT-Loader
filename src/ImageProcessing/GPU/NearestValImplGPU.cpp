#include <Eigen/Dense>
#include <ImageProcessing/GPU/NearestValImplGPU.hpp>
#include <ImageProcessing/GPU/kernels/nearestValue.cuh>

void NearestValImplGPU::runKernel(dim3 threads, dim3 blocks, unsigned char* image, const char channels, float* LUT,
								  const int LUTsize, const float opacity, const std::tuple<int, int>& imgSize) {
	GpuNearestValDevice::run(threads, blocks, image, channels, LUT, LUTsize, opacity, imgSize);
}