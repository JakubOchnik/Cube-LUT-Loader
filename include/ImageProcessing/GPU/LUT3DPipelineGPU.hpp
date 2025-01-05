#pragma once
#include <ImageProcessing/LUTProcessor.hpp>
#include <ImageProcessing/GPU/Utils/CudaCalls.hpp>
#include "device_launch_parameters.h"
#include <tuple>

class LUT3DPipelineGPU : public LUTProcessor {
public:
	LUT3DPipelineGPU(Table3D* lut, CudaCalls* calls);
	cv::Mat execute(cv::Mat img, const float opacity, const uint threadPool) override;

protected:
	virtual void runKernel(dim3 threads, dim3 blocks, unsigned char *image, const char channels, float *LUT, const int LUTsize,
			 const float opacity, const std::tuple<int, int> &imgSize) {}

	Table3D* lut3d;
	CudaCalls* cudaCalls;
};