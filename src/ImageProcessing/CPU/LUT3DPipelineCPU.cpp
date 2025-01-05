#include <ImageProcessing/CPU/LUT3DPipelineCPU.hpp>
#include <ImageProcessing/CPU/WorkerData.hpp>

LUT3DPipelineCPU::LUT3DPipelineCPU(Table3D* lut): lut3d(lut) {}

cv::Mat LUT3DPipelineCPU::execute(cv::Mat img, const float opacity, const uint threadPool) {
	if (!lut3d) {
		throw std::runtime_error("3D LUT is unavailable");
	}
	cv::Mat output = img.clone();
	uchar *image{img.data}, *newImagePtr{output.data};

	// Processing
	// Divide the picture into threadPool vertical windows and process them
	// simultaneously. threadPool - 1 threads will process (WIDTH / threadPool)
	// slices and the last one will process (WIDTH/threadPool +
	// (WIDTH%threadPool))
	const int threadWidth = static_cast<int>(output.cols / threadPool);
	const int remainder = static_cast<int>(output.cols % threadPool);

	// Create a vector of threads to be executed
	std::vector<std::thread> threads;
	threads.reserve(threadPool);

	// Launch threads
	const WorkerData commonData{
		image, newImagePtr, output.cols, output.rows, img.channels(), static_cast<int>(lut3d->dimension(0)), opacity};
	int x{0};
	for (size_t tNum{0}; tNum < threadPool - 1; x += threadWidth, ++tNum) {
		threads.emplace_back([this, x, &commonData, threadWidth]() { calculateArea(x, *lut3d, commonData, threadWidth); });
	}
	// Launch the last thread with a slightly larger width
	const int remainderWidth = threadWidth + remainder;
	threads.emplace_back([this, x, &commonData, remainderWidth]() { calculateArea(x, *lut3d, commonData, remainderWidth); });
	for (auto& thread : threads) {
		thread.join();
	}
	return output;
}
