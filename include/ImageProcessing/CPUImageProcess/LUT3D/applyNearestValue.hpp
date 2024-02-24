#pragma once
#include <DataLoader/CubeLUT.hpp>
#include <opencv2/core.hpp>

struct WorkerData;

namespace NearestValue
{
cv::Mat applyNearest(cv::Mat img, const Table3D& lut, float opacity, uint threadPool);

void calculateArea(int x, const Table3D& lut, float opacity, const WorkerData& data, int segWidth);

void calculatePixel(int x, int y, const Table3D& lut, float opacity, const WorkerData& data);
} // namespace NearestValue
