#include <GPUImageProcess/LUT3D/applyNearestValueHost.hpp>

cv::Mat GpuNearestVal::applyNearestGpu(const Loader& loader, float opacity)
{
	int width{loader.getImg().cols}, height{loader.getImg().rows};
	size_t imgSize = width * height * 3 * sizeof(unsigned char);
	size_t lutSize = pow(loader.getCube().LUT3D.dimension(0), 3) * 3 * sizeof(float);
	float* lutPtr;
	uchar* imgPtr;

	// INIT
	cudaMalloc((void**)&lutPtr, lutSize);
	cudaMemcpy(lutPtr, loader.getCube().LUT3D.data(), lutSize, cudaMemcpyHostToDevice);
	cudaMallocManaged(&imgPtr, imgSize);
	memcpy(imgPtr, loader.getImg().data, imgSize);

	const int threads = 16;
	const int blocksX = (width + threads - 1) / threads;
	const int blocksY = (height + threads - 1) / threads;
	dim3 threadsGrid(threads, threads);
	dim3 blocksGrid(blocksX, blocksY);

	// Process data
	GpuNearestValDevice::run(threadsGrid, blocksGrid, imgPtr, 3, lutPtr, loader.getCube().LUT3D.dimension(0), opacity, std::tuple<int, int>(width, height));
	// Free memory and copy data back to host
	cudaFree(lutPtr);
	cv::Mat finalImg = cv::Mat(height, width, CV_8UC3, imgPtr);
	return finalImg;
}
