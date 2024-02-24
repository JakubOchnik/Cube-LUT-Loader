#pragma once
#include <DataLoader/CubeLUT.hpp>
#include <opencv2/core.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

struct WorkerData;

namespace Trilinear
{
cv::Mat applyTrilinear(cv::Mat img, const Table3D& lut, float opacity, uint threadPool);

void calculateArea(int x, const Table3D& lut, float opacity, const WorkerData& data, int segWidth);

void calculatePixel(int x, int y, const Table3D& lut, float opacity, const WorkerData& data);
} // namespace Trilinear
