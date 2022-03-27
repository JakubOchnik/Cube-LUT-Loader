#include <ImageProcess/LUT3D/applyNearestValue.hpp>

void NearestValue::calculatePixel(const int x, const int y, const CubeLUT& lut, const float opacity,
                                  const WorkerData& data)
{
	const int pixelIndex = (x + y * data.width) * data.channels;

	const int b = data.image[pixelIndex]; // b
	const auto bIdx = static_cast<uint>(round(b * (data.lutSize - 1) / 255.0f));

	const int g = data.image[pixelIndex + 1]; // g
	const auto gIdx = static_cast<uint>(round(g * (data.lutSize - 1) / 255.0f));

	const int r = data.image[pixelIndex + 2]; // r
	const auto rIdx = static_cast<uint>(round(r * (data.lutSize - 1) / 255.0f));

	const auto newB = static_cast<int>(lut.LUT3D(rIdx, gIdx, bIdx, 2) * 255);
	const auto newG = static_cast<int>(lut.LUT3D(rIdx, gIdx, bIdx, 1) * 255);
	const auto newR = static_cast<int>(lut.LUT3D(rIdx, gIdx, bIdx, 0) * 255);

	data.newImage[pixelIndex] = static_cast<uchar>(b + (newB - b) * opacity);
	data.newImage[pixelIndex + 1] = static_cast<uchar>(g + (newG - g) * opacity);
	data.newImage[pixelIndex + 2] = static_cast<uchar>(r + (newR - r) * opacity);
}

void NearestValue::calculateArea(const int x, const CubeLUT& lut, const float opacity, const WorkerData& data,
                                 const int segWidth)
{
	// Iterate over the area of width range: <x, x + segWidth>
	for (int localX{x}; localX < x + segWidth; ++localX)
	{
		for (int y{0}; y < data.height; ++y)
		{
			calculatePixel(localX, y, lut, opacity, data);
		}
	}
}

cv::Mat NearestValue::applyNearest(cv::Mat img, const CubeLUT& lut, const float opacity, const uint threadPool)
{
	// Initialize data
	cv::Mat tmp = img.clone();
	uchar *image{img.data}, *newImage{tmp.data};
	WorkerData commonData{
		image, newImage, tmp.cols, tmp.rows, img.channels(),
		static_cast<int>(lut.LUT3D.dimension(0))
	};

	// Processing
	// Divide the picture into threadPool vertical windows and process them simultaneously.
	// threadPool - 1 threads will process (WIDTH / threadPool) slices
	// and the last one will process (WIDTH/threadPool + (WIDTH%threadPool))
	const int threadWidth = static_cast<int>(tmp.cols / threadPool);
	const int remainder = static_cast<int>(tmp.cols % threadPool);

	// Create a vector of threads to be executed
	std::vector<std::thread> threads;
	threads.reserve(threadPool);

	// Launch threads
	int x{0};
	for (size_t tNum{0}; tNum < threadPool - 1; x += threadWidth, ++tNum)
	{
		threads.emplace_back(calculateArea, x, std::cref(lut), opacity, std::ref(commonData), threadWidth);
	}
	// Launch the last thread with a slightly larger width
	threads.emplace_back(calculateArea, x, std::cref(lut), opacity, std::ref(commonData), threadWidth + remainder);
	for (auto& thread : threads)
	{
		thread.join();
	}
	return tmp;
}
