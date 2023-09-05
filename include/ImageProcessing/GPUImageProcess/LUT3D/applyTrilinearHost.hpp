#pragma once
#include <DataLoader/DataLoader.hpp>
#include <opencv2/opencv.hpp>

namespace GpuTrilinear
{
cv::Mat applyTrilinearGpu(const DataLoader& loader, float opacity, int threads);
}
