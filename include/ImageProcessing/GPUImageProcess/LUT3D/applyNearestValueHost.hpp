#pragma once
#include <DataLoader/DataLoader.hpp>
#include <opencv2/core.hpp>

namespace GpuNearestVal
{
cv::Mat applyNearestGpu(const DataLoader& loader, float opacity, int threads);
}
