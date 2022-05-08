#pragma once
#include <opencv2/opencv.hpp>

#include <DataLoader/dataLoader.hpp>

namespace GpuNearestVal
{
	cv::Mat applyNearestGpu(const DataLoader& loader, float opacity, int threads);
}
