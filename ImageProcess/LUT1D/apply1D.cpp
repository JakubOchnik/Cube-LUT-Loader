#include "apply1D.hpp"

double getAvgVal(const CubeLUT& lut, const unsigned int nValues, const unsigned char value, const unsigned char channel)
{
	double val{ 0.0 };
	for (unsigned int i{ 0 }; i < nValues; ++i)
	{
		val += lut.LUT1D[value * nValues + i][channel];
	}
	return val / nValues;
}

unsigned char getColor(const double value)
{
	if (value > 1.0) // assuming that max domain is 1.0
	{
		return 255;
	}
	return static_cast<unsigned char>(round(value * 255));
}

cv::Mat_<cv::Vec3b> applyBasic1D(const cv::Mat& img, const CubeLUT& lut, const float opacity)
{
	cv::Mat_<cv::Vec3b> tmp = img.clone();
	const unsigned int lutSize{ static_cast<unsigned int>(lut.LUT1D.size()) };
	const unsigned int nValues{ lutSize / 256 }; //assuming 8-bit image

	for (auto& pixel : tmp)
	{

		const unsigned char newB = getColor(getAvgVal(lut, nValues, pixel[0], 2)); // b
		const unsigned char newG = getColor(getAvgVal(lut, nValues, pixel[1], 1)); // g
		const unsigned char newR = getColor(getAvgVal(lut, nValues, pixel[2], 0)); // r

		const unsigned char finalB = pixel[0] + (newB - pixel[0]) * opacity;
		const unsigned char finalG = pixel[1] + (newG - pixel[1]) * opacity;
		const unsigned char finalR = pixel[2] + (newR - pixel[2]) * opacity;

		pixel[0] = finalB;
		pixel[1] = finalG;
		pixel[2] = finalR;
	}
	return tmp;
}