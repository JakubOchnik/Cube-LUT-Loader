#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ImageProcessing/GPU/LUT3DPipelineGPU.hpp>
#include <ImageProcessing/GPU/Utils/CudaUtils.hpp>
#include <tuple>

cv::Mat LUT3DPipelineGPU::execute(cv::Mat img, const float opacity, const uint threadPool) {
	if (!lut3d) {
		throw std::runtime_error("3D LUT is unavailable");
	}
	int width{img.cols}, height{img.rows};
	const int imgSize = width * height * 3 * sizeof(unsigned char);
	const int lutSize = static_cast<int>(pow(lut3d->dimension(0), 3) * 3 * sizeof(float));

	// Declare device (or/and host) pointers
	float* lutPtr{nullptr};
	uchar* imgPtr{nullptr};

	// Copy data to GPU
	cudaErrorChk(cudaMalloc(reinterpret_cast<void**>(&lutPtr), lutSize));
	cudaErrorChk(cudaMemcpy(lutPtr, lut3d->data(), lutSize, cudaMemcpyHostToDevice));
	cudaErrorChk(cudaMallocManaged(&imgPtr, imgSize));
	memcpy(imgPtr, img.data, imgSize);

	const int blocksX = (width + threadPool - 1) / threadPool;
	const int blocksY = (height + threadPool - 1) / threadPool;
	const dim3 threadsGrid(threadPool, threadPool);
	const dim3 blocksGrid(blocksX, blocksY);

	// Process data
	runKernel(threadsGrid, blocksGrid, imgPtr, 3, lutPtr, lut3d->dimension(0), opacity, {width, height});

	// Free memory and copy data back to host
	cudaErrorChk(cudaFree(lutPtr));
	auto finalImg = cv::Mat(height, width, CV_8UC3, imgPtr);
	return finalImg;
}
