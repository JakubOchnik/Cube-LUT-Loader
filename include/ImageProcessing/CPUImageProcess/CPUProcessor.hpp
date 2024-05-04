#pragma once
#include <FileIO/FileIO.hpp>
#include <opencv2/core.hpp>
#include <ImageProcessing/ImageProcessor.hpp>

class CPUProcessor : public ImageProcessor
{
	uint numberOfThreads{1};
public:
	explicit CPUProcessor(FileIO& fileIfc, uint threads);
	cv::Mat process(float strength, InterpolationMethod method) override;
};
