#include <ImageProcessing/CPU/NearestValImplCPU.hpp>
#include <ImageProcessing/CPU/WorkerData.hpp>
#include <functional>
#include <thread>
#include <vector>

void NearestValImplCPU::calculatePixel(const int x, const int y, const Table3D& lut, const WorkerData& data) {
	const int pixelIndex = (x + y * data.width) * data.channels;

	const int b = data.image[pixelIndex]; // b
	const auto bIdx = static_cast<uint>(round(b * (data.lutSize - 1) / 255.0f));

	const int g = data.image[pixelIndex + 1]; // g
	const auto gIdx = static_cast<uint>(round(g * (data.lutSize - 1) / 255.0f));

	const int r = data.image[pixelIndex + 2]; // r
	const auto rIdx = static_cast<uint>(round(r * (data.lutSize - 1) / 255.0f));

	const auto newB = static_cast<int>(lut(rIdx, gIdx, bIdx, 2) * 255);
	const auto newG = static_cast<int>(lut(rIdx, gIdx, bIdx, 1) * 255);
	const auto newR = static_cast<int>(lut(rIdx, gIdx, bIdx, 0) * 255);

	data.newImage[pixelIndex] = static_cast<uchar>(b + (newB - b) * data.opacity);
	data.newImage[pixelIndex + 1] = static_cast<uchar>(g + (newG - g) * data.opacity);
	data.newImage[pixelIndex + 2] = static_cast<uchar>(r + (newR - r) * data.opacity);
}
