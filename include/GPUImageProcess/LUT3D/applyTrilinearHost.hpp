#pragma once
#include <opencv2/opencv.hpp>

#include <DataLoader/dataLoader.hpp>

namespace GpuTrilinear
{
	cv::Mat applyTrilinearGpu(const DataLoader& loader, float opacity, int threads);
}
