#pragma once
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <tuple>

#include <DataLoader/CubeLUT.hpp>
#include <DataLoader/dataLoader.hpp>
#include <GPUImageProcess/LUT3D/applyNearestValueGpu.cuh>
#include <GPUImageProcess/Utils/cudaUtils.hpp>

namespace GpuNearestVal
{
	cv::Mat applyNearestGpu(const DataLoader& loader, float opacity, int threads);
}
