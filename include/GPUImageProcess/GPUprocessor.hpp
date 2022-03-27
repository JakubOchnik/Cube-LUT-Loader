#pragma once
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <Loader/CubeLUT.hpp>
#include <Loader/loader.hpp>

#include <GPUImageProcess/LUT3D/applyNearestValueHost.hpp>
#include <GPUImageProcess/LUT3D/applyTrilinearHost.hpp>
#include <GPUImageProcess/Utils/CudaUtils.hpp>


class GpuProcessor
{
	cv::Mat newImg;
	const Loader& loader;
public:
	GpuProcessor(const Loader& ld);
	GpuProcessor() = delete;
	cv::Mat process();
	void save() const;
	void setLoader(const Loader& ld) const;
	void perform();
};
