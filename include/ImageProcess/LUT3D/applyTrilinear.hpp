#pragma once
#include <opencv2/opencv.hpp>
#include <Loader/CubeLUT.hpp>
#include <Eigen/Dense>
#include <thread>
#include <vector>
#include <functional>

namespace Trilinear
{
	struct WorkerData
	{
		unsigned char* image;
		unsigned char* new_image;
		const int width;
		const int height;
		const int channels;
		const int lutSize;
		const Eigen::array<Eigen::Index, 4> extents;
	};

	cv::Mat applyTrilinear(cv::Mat img, const CubeLUT& lut, float opacity, const uint threadPool);
	void calculateArea(const int x, const CubeLUT& lut, const float opacity, WorkerData& data, const int segWidth);
	void calculatePixel(const int x, const int y, const CubeLUT& lut, const float opacity, WorkerData& data);
}