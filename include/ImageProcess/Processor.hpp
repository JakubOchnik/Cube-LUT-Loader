#pragma once
#include <opencv2/opencv.hpp>
#include <DataLoader/dataLoader.hpp>

class Processor
{
	cv::Mat_<cv::Vec3b> newImg;
	const DataLoader& loader;
public:
	Processor(const DataLoader& ld);
	Processor() = delete;
	cv::Mat_<cv::Vec3b> process();
	void save() const;
	void execute();
};
