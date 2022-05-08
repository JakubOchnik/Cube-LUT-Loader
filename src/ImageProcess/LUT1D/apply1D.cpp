#include <ImageProcess/LUT1D/apply1D.hpp>
#include <thread>
#include <vector>

float Basic1D::getAvgVal(const CubeLUT& lut, const uint nValues, const uchar value, const uchar channel)
{
	// Cast to avoid potential index overflow
	using EigenIndex = Eigen::Tensor<float, 2, 0, Eigen::DenseIndex>::Index;

	float val{0.0f};
	for (uint i{0}; i < nValues; ++i)
	{
		val += lut.LUT1D(value * static_cast<EigenIndex>(nValues) + i, channel);
	}
	return val / static_cast<float>(nValues);
}

uchar Basic1D::getClippedVal(const float value)
{
	// Assuming that max domain is 1.0. TODO
	if (value > 1.0f)
	{
		return 255;
	}
	return static_cast<uchar>(round(value * 255));
}

void Basic1D::calculatePixel(const int x, const int y, const CubeLUT& lut, const float opacity, const WorkerData& data)
{
	const uchar b = data.image[(x + y * data.width) * data.channels + 0]; // b
	const uchar g = data.image[(x + y * data.width) * data.channels + 1]; // g
	const uchar r = data.image[(x + y * data.width) * data.channels + 2]; // r

	const uchar newB = getClippedVal(getAvgVal(lut, data.nValues, b, 2)); // b
	const uchar newG = getClippedVal(getAvgVal(lut, data.nValues, g, 1)); // g
	const uchar newR = getClippedVal(getAvgVal(lut, data.nValues, r, 0)); // r

	data.newImage[(x + y * data.width) * data.channels + 0] = b + static_cast<uchar>((newB - b) * opacity);
	data.newImage[(x + y * data.width) * data.channels + 1] = g + static_cast<uchar>((newG - g) * opacity);
	data.newImage[(x + y * data.width) * data.channels + 2] = r + static_cast<uchar>((newR - r) * opacity);
}

void Basic1D::calculateArea(const int x, const CubeLUT& lut, const float opacity, const WorkerData& data,
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

cv::Mat_<cv::Vec3b> Basic1D::applyBasic1D(const cv::Mat& img, const CubeLUT& lut, const float opacity,
                                          const uint threadPool)
{
	cv::Mat_<cv::Vec3b> tmp = img.clone();
	uchar *image{img.data}, *newImage{tmp.data};

	const int lutSize{static_cast<int>(lut.LUT1D.dimension(0) / 3)};
	const int nValues{lutSize / 256}; //assuming 8-bit image
	WorkerData commonData{
		image, newImage, tmp.cols, tmp.rows, img.channels(),
		lutSize, nValues
	};

	const int threadWidth = static_cast<int>(tmp.cols / threadPool);
	const int remainder = static_cast<int>(tmp.cols % threadPool);

	// Create a vector of threads to be executed
	std::vector<std::thread> threads;
	threads.reserve(threadPool);

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
