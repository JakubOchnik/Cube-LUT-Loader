#pragma once
#include <FileIO/FileIO.hpp>
#include <ImageProcessing/ImageProcessExecutor.hpp>
#include <opencv2/core.hpp>

class CPUModeExecutor : public ImageProcessExecutor {
	uint numberOfThreads{1};

public:
	explicit CPUModeExecutor(FileIO& fileIfc, uint threads);
	cv::Mat process(float intensity, InterpolationMethod method) override;
};
