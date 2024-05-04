#pragma once
#include <FileIO/FileIO.hpp>
#include <opencv2/core.hpp>
#include <ImageProcessing/ImageProcessor.hpp>

class CPUProcessor : public ImageProcessor
{
public:
	explicit CPUProcessor(const FileIO& ld);
	cv::Mat process() override;
};
