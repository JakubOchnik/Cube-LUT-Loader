#pragma once
#include <FileIO/FileIO.hpp>
#include <opencv2/core.hpp>
#include <ImageProcessing/ImageProcessExecutor.hpp>
#include <ImageProcessing/GPU/Utils/CudaCalls.hpp>

class GPUModeExecutor : public ImageProcessExecutor
{
	static constexpr int threadsPerBlock{16};

public:
	GPUModeExecutor(FileIO& fileIfc);
	~GPUModeExecutor();

private:
	cv::Mat process(float intensity, InterpolationMethod method) override;

	CudaCalls calls;
};
