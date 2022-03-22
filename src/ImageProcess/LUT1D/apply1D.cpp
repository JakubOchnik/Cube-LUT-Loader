#include <ImageProcess/LUT1D/apply1D.hpp>

double Basic1D::getAvgVal(const CubeLUT& lut, const uint nValues, const uchar value, const uchar channel)
{
	double val{ 0.0 };
	for (uint i{ 0 }; i < nValues; ++i)
	{
		val += lut.LUT1D(value * nValues + i, channel);
	}
	return val / nValues;
}

uchar Basic1D::getColor(const double value)
{
	if (value > 1.0) // assuming that max domain is 1.0
	{
		return 255;
	}
	return static_cast<uchar>(round(value * 255));
}

void Basic1D::calculatePixel(const int x, const int y, const CubeLUT& lut, const float opacity, WorkerData& data)
{
	int b = data.image[(x + y * data.width) * data.channels + 0]; // b
	int g = data.image[(x + y * data.width) * data.channels + 1]; // g
	int r = data.image[(x + y * data.width) * data.channels + 2]; // r

	const uchar newB = getColor(getAvgVal(lut, data.nValues, b, 2)); // b
	const uchar newG = getColor(getAvgVal(lut, data.nValues, g, 1)); // g
	const uchar newR = getColor(getAvgVal(lut, data.nValues, r, 0)); // r

	const uchar finalB = b + (newB - b) * opacity;
	const uchar finalG = g + (newG - g) * opacity;
	const uchar finalR = r + (newR - r) * opacity;

	data.new_image[(x + y * data.width) * data.channels + 0] = finalB;
	data.new_image[(x + y * data.width) * data.channels + 1] = finalG;
	data.new_image[(x + y * data.width) * data.channels + 2] = finalR;
}

void Basic1D::calculateArea(const int x, const CubeLUT& lut, const float opacity, WorkerData& data, const int segWidth)
{
	for(int localX{x}; localX < x + segWidth; ++localX)
	{
		for(int y{0}; y < data.height; ++y)
		{
			calculatePixel(localX, y, lut, opacity, data);
		}
	}
}

cv::Mat_<cv::Vec3b> Basic1D::applyBasic1D(const cv::Mat& img, const CubeLUT& lut, const float opacity, const uint threadPool)
{
	cv::Mat_<cv::Vec3b> tmp = img.clone();
	unsigned char* image = img.data;
	unsigned char* new_image = tmp.data;

	const int lutSize{ static_cast<int>(lut.LUT1D.dimension(0) / 3) };
	const int nValues{ lutSize / 256 }; //assuming 8-bit image
	WorkerData commonData{image, new_image, tmp.cols, tmp.rows, img.channels(), \
							lutSize, nValues};

	int threadWidth = tmp.cols / threadPool;
	int remainder = tmp.cols % threadPool;
	std::vector<std::thread> threads;
	threads.reserve(threadPool);

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