#include <ImageProcessing/CPU/Simple1DImplCPU.hpp>
#include <ImageProcessing/CPU/WorkerData.hpp>
#include <thread>
#include <vector>

Simple1DImplCPU::Simple1DImplCPU(Table1D* lut) : lut1d(lut) {}

namespace {
float getAvgVal(const Table1D& lut, const uint nValues, const uchar value, const uchar channel) {
	// Cast to avoid potential index overflow
	using EigenIndex = Eigen::Tensor<float, 2, 0, Eigen::DenseIndex>::Index;

	float val{0.0f};
	for (uint i{0}; i < nValues; ++i) {
		val += lut(value * static_cast<EigenIndex>(nValues) + i, channel);
	}
	return val / static_cast<float>(nValues);
}

uchar getClippedVal(const float value) {
	// Assuming that max domain is 1.0. TODO
	if (value > 1.0f) {
		return 255;
	}
	return static_cast<uchar>(round(value * 255));
}
} // namespace

void Simple1DImplCPU::calculatePixel(const int x, const int y, const Table1D& lut, const WorkerData& data) {
	const size_t pixelIndex = (x + y * data.width) * data.channels;
	const uchar b = data.image[pixelIndex + 0]; // b
	const uchar g = data.image[pixelIndex + 1]; // g
	const uchar r = data.image[pixelIndex + 2]; // r

	const uchar newB = getClippedVal(getAvgVal(lut, data.nValues, b, 2)); // b
	const uchar newG = getClippedVal(getAvgVal(lut, data.nValues, g, 1)); // g
	const uchar newR = getClippedVal(getAvgVal(lut, data.nValues, r, 0)); // r

	data.newImage[pixelIndex + 0] = b + static_cast<uchar>((newB - b) * data.opacity);
	data.newImage[pixelIndex + 1] = g + static_cast<uchar>((newG - g) * data.opacity);
	data.newImage[pixelIndex + 2] = r + static_cast<uchar>((newR - r) * data.opacity);
}

void Simple1DImplCPU::calculateArea(const int x, const Table1D& lut, const WorkerData& data, const int segWidth) {
	// Iterate over the area of width range: <x, x + segWidth>
	for (int localX{x}; localX < x + segWidth; ++localX) {
		for (int y{0}; y < data.height; ++y) {
			calculatePixel(localX, y, lut, data);
		}
	}
}

cv::Mat Simple1DImplCPU::execute(cv::Mat img, const float opacity, const uint threadPool) {
	if (!lut1d) {
		throw std::runtime_error("1D LUT is unavailable");
	}
	cv::Mat tmp = img.clone();
	uchar *image{img.data}, *newImage{tmp.data};

	const int lutSize{static_cast<int>(lut1d->dimension(0) / 3)};
	const int nValues{lutSize / 256}; // assuming 8-bit image

	const int threadWidth = static_cast<int>(tmp.cols / threadPool);
	const int remainder = static_cast<int>(tmp.cols % threadPool);

	// Create a vector of threads to be executed
	std::vector<std::thread> threads;
	threads.reserve(threadPool);

	const WorkerData commonData{image, newImage, tmp.cols, tmp.rows, img.channels(), lutSize, opacity, nValues};
	int x{0};
	for (size_t tNum{0}; tNum < threadPool - 1; x += threadWidth, ++tNum) {
		threads.emplace_back([this, x, &commonData, threadWidth]() { calculateArea(x, *lut1d, commonData, threadWidth); });
	}
	// Launch the last thread with a slightly larger width
	const int remainderWidth = threadWidth + remainder;
	threads.emplace_back([this, x, &commonData, remainderWidth]() { calculateArea(x, *lut1d, commonData, remainderWidth); });
	for (auto& thread : threads) {
		thread.join();
	}
	return tmp;
}
