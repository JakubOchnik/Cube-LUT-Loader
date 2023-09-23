#pragma once
#include <DataLoader/CubeLUT.hpp>
#include <opencv2/core.hpp>

namespace Basic1D
{
struct WorkerData
{
	unsigned char* image;
	unsigned char* newImage;
	const int	   width;
	const int	   height;
	const int	   channels;
	const int	   lutSize;
	const int	   nValues;
};

cv::Mat_<cv::Vec3b> applyBasic1D(const cv::Mat& img, const Table1D& lut, float opacity, uint threadPool);

void calculateArea(int x,const Table1D& lut, float opacity, const WorkerData& data, int segWidth);

void calculatePixel(int x, int y, const Table1D& lut, float opacity, const WorkerData& data);

float getAvgVal(const Table1D& lut, uint nValues, uchar value, uchar channel);

uchar getClippedVal(float value);
} // namespace Basic1D
