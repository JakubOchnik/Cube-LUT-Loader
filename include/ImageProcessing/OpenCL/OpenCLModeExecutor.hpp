#pragma once

#include "FileIO/FileIO.hpp"
#include "ImageProcessing/OpenCL/Utils/OpenRuntime.hpp"
#include <ImageProcessing/ImageProcessExecutor.hpp>

class OpenCLModeExecutor : public ImageProcessExecutor {
	static constexpr int workGroups{16};

public:
	OpenCLModeExecutor(FileIO& fileIfc);
	~OpenCLModeExecutor() = default;
	void initialize() { runtime.initialize(); }

private:
	cv::Mat process(float intensity, InterpolationMethod method) override;
	OpenCLRuntime runtime;
};
