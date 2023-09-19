#pragma once
#include <DataLoader/DataLoader.hpp>
#include <opencv2/core.hpp>
#include <ImageProcessing/ImageProcessor.hpp>

class CPUProcessor : public ImageProcessor
{
public:
	CPUProcessor(const DataLoader &ld);
	cv::Mat process() override;
};
