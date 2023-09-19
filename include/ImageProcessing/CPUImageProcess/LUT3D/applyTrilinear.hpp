#pragma once
#include <DataLoader/CubeLUT.hpp>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace Trilinear
{
struct WorkerData
{
	unsigned char*						image;
	unsigned char*						newImage;
	const int							width;
	const int							height;
	const int							channels;
	const int							lutSize;
	const Eigen::array<Eigen::Index, 4> extents;
};

cv::Mat applyTrilinear(cv::Mat img, const CubeLUT& lut, float opacity, uint threadPool);

void calculateArea(int				 x,
				   const CubeLUT&	 lut,
				   float			 opacity,
				   const WorkerData& data,
				   int				 segWidth);

void calculatePixel(
	int x, int y, const CubeLUT& lut, float opacity, const WorkerData& data);
} // namespace Trilinear
