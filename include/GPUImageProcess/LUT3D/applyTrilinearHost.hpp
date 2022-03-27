#pragma once
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <tuple>

#include <Loader/CubeLUT.hpp>
#include <Loader/Loader.hpp>

#include <GPUImageProcess/MultidimData/MultidimDataUtils.hpp>
#include <GPUImageProcess/LUT3D/applyTrilinearGpu.cuh>

namespace GpuTrilinear
{
    cv::Mat applyTrilinearGpu(const Loader& loader, float opacity);
}
