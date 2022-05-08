#pragma once
#include <DataLoader/dataLoader.hpp>
#include <opencv2/opencv.hpp>

namespace GpuNearestVal
{
cv::Mat applyNearestGpu(const DataLoader& loader, float opacity, int threads);
}
