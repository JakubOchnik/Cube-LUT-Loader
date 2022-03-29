#pragma once
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <DataLoader/CubeLUT.hpp>
#include <DataLoader/dataLoader.hpp>
#include <GPUImageProcess/LUT3D/applyNearestValueHost.hpp>
#include <GPUImageProcess/LUT3D/applyTrilinearHost.hpp>
#include <GPUImageProcess/Utils/cudaUtils.hpp>


class GpuProcessor
{
	const int threadsPerBlock{16};
	cv::Mat newImg;
	const DataLoader& loader;
public:
	GpuProcessor(const DataLoader& ld);
	GpuProcessor() = delete;
	cv::Mat process();
	void save() const;
	void execute();
};
