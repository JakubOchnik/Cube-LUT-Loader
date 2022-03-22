#include <ImageProcess/LUT3D/applyNearestValue.hpp>

void NearestValue::calculatePixel(const int x, const int y, const CubeLUT& lut, const float opacity, WorkerData& data)
{
	int b = data.image[(x + y * data.width) * data.channels + 0]; // b
	unsigned int b_ind = round(b * (data.lutSize - 1) / 255.0f);
	int g = data.image[(x + y * data.width) * data.channels + 1]; // g
	unsigned int g_ind = round(g * (data.lutSize - 1) / 255.0f);
	int r = data.image[(x + y * data.width) * data.channels + 2]; // r
	unsigned int r_ind = round(r * (data.lutSize - 1) / 255.0f);

	int newB = static_cast<int>(lut.LUT3D(r_ind, g_ind, b_ind, 2) * 255);
	int newG = static_cast<int>(lut.LUT3D(r_ind, g_ind, b_ind, 1) * 255);
	int newR = static_cast<int>(lut.LUT3D(r_ind, g_ind, b_ind, 0) * 255);

	unsigned char finalB = b + (newB - b) * opacity;
	unsigned char finalG = g + (newG - g) * opacity;
	unsigned char finalR = r + (newR - r) * opacity;

	data.new_image[(x + y * data.width) * data.channels + 0] = finalB;
	data.new_image[(x + y * data.width) * data.channels + 1] = finalG;
	data.new_image[(x + y * data.width) * data.channels + 2] = finalR;
}

void NearestValue::calculateArea(const int x, const CubeLUT& lut, const float opacity, WorkerData& data, const int segWidth)
{
	for(int localX{x}; localX < x + segWidth; ++localX)
	{
		for(int y{0}; y < data.height; ++y)
		{
			calculatePixel(localX, y, lut, opacity, data);
		}
	}
}

cv::Mat NearestValue::applyNearest(cv::Mat img, const CubeLUT& lut, const float opacity, const uint threadPool)
{
	// INIT
	cv::Mat tmp = img.clone();
	unsigned char* image = img.data;
	unsigned char* new_image = tmp.data;
	WorkerData commonData{image, new_image, tmp.cols, tmp.rows, img.channels(), \
							static_cast<int>(lut.LUT3D.dimension(0))};

	// Processing
	// Divide the picture into threadPool vertical windows and process them simultaneously.
	// threadPool - 1 threads will process (WIDTH / threadPool) slices
	// and the last one will process (WIDTH/threadPool + (WIDTH%threadPool))

	int threadWidth = tmp.cols / threadPool;
	int remainder = tmp.cols % threadPool;
	std::vector<std::thread> threads;
	threads.reserve(threadPool);

	// Launch threads
	int x{0}, tNum{0};
	for (; tNum < threadPool - 1; x += threadWidth, ++tNum)
	{
		threads.emplace_back(calculateArea, x, std::cref(lut), opacity, std::ref(commonData), threadWidth);
	}
	// Launch the last thread with a slightly larger width
	threads.emplace_back(calculateArea, x, std::cref(lut), opacity, std::ref(commonData), threadWidth + remainder);
	for(auto& thread: threads)
	{
		thread.join();
	}
	return tmp;
}
