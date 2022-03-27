#pragma once
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

#include <DataLoader/CubeLUT.hpp>
#include <DataLoader/dataLoader.hpp>
#include <ImageProcess/LUT1D/apply1D.hpp>
#include <ImageProcess/LUT3D/applyTrilinear.hpp>
#include <ImageProcess/LUT3D/applyNearestValue.hpp>


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
