#include <ImageProcessing/GPU/TrilinearImplGPU.hpp>
#include <ImageProcessing/GPU/kernels/trilinear.cuh>

void TrilinearImplGPU::runKernel(dim3 threads, dim3 blocks, unsigned char* image, const char channels, float* LUT,
								 const int LUTsize, const float opacity, const std::tuple<int, int>& imgSize) {
	GpuTrilinearDevice::run(threads, blocks, image, channels, LUT, LUTsize, opacity, imgSize);
}