#pragma once
#include <DataLoader/DataLoader.hpp>
#include <opencv2/opencv.hpp>

namespace GpuNearestVal
{
cv::Mat applyNearestGpu(const DataLoader& loader, float opacity, int threads);
}
