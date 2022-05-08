#pragma once
#include <DataLoader/dataLoader.hpp>
#include <opencv2/opencv.hpp>

namespace GpuTrilinear
{
cv::Mat applyTrilinearGpu(const DataLoader& loader, float opacity, int threads);
}
