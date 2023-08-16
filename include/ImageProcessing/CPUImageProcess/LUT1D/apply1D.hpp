#pragma once
#include <DataLoader/CubeLUT.hpp>
#include <opencv2/opencv.hpp>

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

cv::Mat_<cv::Vec3b> applyBasic1D(const cv::Mat& img,
								 const CubeLUT& lut,
								 float			opacity,
								 uint			threadPool);

void calculateArea(int				 x,
				   const CubeLUT&	 lut,
				   float			 opacity,
				   const WorkerData& data,
				   int				 segWidth);

void calculatePixel(
	int x, int y, const CubeLUT& lut, float opacity, const WorkerData& data);

float getAvgVal(const CubeLUT& lut, uint nValues, uchar value, uchar channel);

uchar getClippedVal(float value);
} // namespace Basic1D
