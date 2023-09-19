#pragma once
#include <DataLoader/DataLoader.hpp>
#include <opencv2/core.hpp>

namespace GpuTrilinear
{
cv::Mat applyTrilinearGpu(const DataLoader& loader, float opacity, int threads);
}
