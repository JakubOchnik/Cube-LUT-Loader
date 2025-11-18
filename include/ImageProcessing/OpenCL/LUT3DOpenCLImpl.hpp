
#pragma once
#include "ImageProcessing/OpenCL/OpenCLModeExecutor.hpp"
#include "TaskDispatcher/InputParams.h"
#include <CL/cl.h>
#include <ImageProcessing/LUTProcessor.hpp>

class LUT3DOpenCLImpl : public LUTProcessor {
public:
	LUT3DOpenCLImpl(Table3D* lut, OpenCLRuntime* OCLRuntime, InterpolationMethod interpolationMethod);
	cv::Mat execute(cv::Mat img, const float opacity, const uint threadPool) override;

private:
	[[nodiscard]]
	std::vector<float> prepareLUTImage() const;
	cl_image_format getLUTFormat() const;
  Table3D* lut3d;
  OpenCLRuntime* runtime;
  InterpolationMethod method;
};
