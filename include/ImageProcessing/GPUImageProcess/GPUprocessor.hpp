#pragma once
#include <DataLoader/dataLoader.hpp>
#include <opencv2/opencv.hpp>
#include <ImageProcessing/ImageProcessor.hpp>

class GpuProcessor : public ImageProcessor
{
	static constexpr int threadsPerBlock{16};

public:
	GpuProcessor(const DataLoader &ld);
	~GpuProcessor();
	void execute() override;

private:
	cv::Mat process() override;
	bool isCudaAvailable() const;
};
