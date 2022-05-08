#pragma once
#include <opencv2/opencv.hpp>
#include <DataLoader/dataLoader.hpp>

class GpuProcessor
{
	const int threadsPerBlock{16};
	cv::Mat newImg;
	const DataLoader& loader;
public:
	GpuProcessor(const DataLoader& ld);
	GpuProcessor() = delete;
	void execute();
private:
	cv::Mat process();
	void save() const;
	bool isCudaAvailable() const;
};
