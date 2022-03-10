#pragma once
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <Loader/CubeLUT.hpp>
#include <Loader/loader.hpp>

#include <ImageProcess/LUT1D/apply1D.hpp>
#include <ImageProcess/LUT3D/applyTrilinear.hpp>
#include <ImageProcess/LUT3D/applyNearestValue.hpp>


class Processor
{
	cv::Mat_<cv::Vec3b> newImg;
	const Loader& loader;
public:
	Processor(const Loader& ld);
	Processor() = delete;
	bool is3D() const;
	cv::Mat_<cv::Vec3b> process();
	void save() const;
	void setLoader(const Loader& ld) const;
	void perform();
};
