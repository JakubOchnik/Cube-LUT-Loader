#pragma once
#include <FileIO/FileIO.hpp>
#include <opencv2/core.hpp>
#include <ImageProcessing/ImageProcessExecutor.hpp>

class GpuProcessor : public ImageProcessExecutor
{
	static constexpr int threadsPerBlock{16};

public:
	GpuProcessor(FileIO& fileIfc);
	~GpuProcessor();

private:
	cv::Mat process(float strength, InterpolationMethod method) override;
};
