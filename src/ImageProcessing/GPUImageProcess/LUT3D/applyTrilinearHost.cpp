#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/GPUImageProcess/LUT3D/applyTrilinearGpu.cuh>
#include <ImageProcessing/GPUImageProcess/LUT3D/applyTrilinearHost.hpp>
#include <ImageProcessing/GPUImageProcess/Utils/CudaUtils.hpp>
#include <tuple>

cv::Mat GpuTrilinear::applyTrilinearGpu(cv::Mat input, const Table3D &lut, const float opacity, const int threads)
{
	int width{input.cols}, height{input.rows};
	const int imgSize = width * height * 3 * sizeof(unsigned char);
	const int lutSize = static_cast<int>(
		pow(lut.dimension(0), 3) * 3 * sizeof(float));

	// Declare device (or/and host) pointers
	float* lutPtr{nullptr};
	uchar* imgPtr{nullptr};

	// Copy data to GPU
	cudaErrorChk(cudaMalloc(reinterpret_cast<void**>(&lutPtr), lutSize));
	cudaErrorChk(cudaMemcpy(lutPtr, lut.data(), lutSize, cudaMemcpyHostToDevice));
	cudaErrorChk(cudaMallocManaged(&imgPtr, imgSize));
	memcpy(imgPtr, input.data, imgSize);

	const int blocksX = (width + threads - 1) / threads;
	const int blocksY = (height + threads - 1) / threads;
	const dim3 threadsGrid(threads, threads);
	const dim3 blocksGrid(blocksX, blocksY);

	// Process data
	GpuTrilinearDevice::run(threadsGrid, blocksGrid, imgPtr, 3, lutPtr, lut.dimension(0), opacity, {width, height});

	// Free memory and copy data back to host
	cudaErrorChk(cudaFree(lutPtr));
	auto finalImg = cv::Mat(height, width, CV_8UC3, imgPtr);
	return finalImg;
}
