#include "applyNearestValueHost.hpp"

cv::Mat applyNearestGpu(const Loader& loader, float opacity)
{
	int width = loader.getImg().cols, height = loader.getImg().rows;
	size_t imgSize = width * height * 3 * sizeof(unsigned char);
	size_t lutSize = pow(loader.getCube().LUT3D.size(), 3) * 3 * sizeof(float);

	std::vector<float> flattenedLUT;
	float* lutPtr;
	unsigned char* imgPtr;
	// INIT
	cudaMallocManaged(&lutPtr, lutSize);
	cudaMallocManaged(&imgPtr, imgSize);

	flattenedLUT = flatten4D<float>(loader.getCube().LUT3D);
	std::vector<float>& flatLut = flattenedLUT;
	memcpy(lutPtr, flattenedLUT.data(), lutSize);
	memcpy(imgPtr, loader.getImg().data, imgSize);

	dim3 grid(width, height);
	// PROCESS
	NearestValGpu::run(grid, imgPtr, 3, lutPtr, loader.getCube().LUT3D.size(), opacity);
	// CLEANUP1
	cudaFree(lutPtr);

	cv::Mat finalImg = cv::Mat(height, width, CV_8UC3, imgPtr);
	return finalImg;
}
