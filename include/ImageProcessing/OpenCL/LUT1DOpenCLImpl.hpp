
#pragma once
#include "FileIO/CubeLUT.hpp"
#include "ImageProcessing/OpenCL/OpenCLModeExecutor.hpp"
#include <ImageProcessing/LUTProcessor.hpp>

class LUT1DImplOpenCL : public LUTProcessor {
public:
	LUT1DImplOpenCL(Table1D* lut, OpenCLRuntime* OCLRuntime);
	cv::Mat execute(cv::Mat img, const float opacity, const uint threadPool) override;

private:
	[[nodiscard]]
	std::vector<float> prepareLUTImage() const;
	cl_image_format getLUTFormat() const;
  Table1D* lut1d;
  OpenCLRuntime* runtime;
};
