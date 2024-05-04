#pragma once
#include <FileIO/FileIO.hpp>
#include <opencv2/core.hpp>
#include <ImageProcessing/ImageProcessor.hpp>

class GpuProcessor : public ImageProcessor
{
	static constexpr int threadsPerBlock{16};

public:
	GpuProcessor(const FileIO &ld);
	~GpuProcessor();
	void execute() override;

private:
	cv::Mat process() override;
	bool isCudaAvailable() const;
};
