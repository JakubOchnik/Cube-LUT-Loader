#pragma once
#include <boost/program_options.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <Loader/CubeLUT.hpp>
#include <fstream>

class Loader
{
	cv::Mat_<cv::Vec3b> img;
	CubeLUT cube;
	boost::program_options::variables_map vm;

public:
	Loader(boost::program_options::variables_map vm);
	Loader();
	void setArgs(boost::program_options::variables_map vm);
	void loadImg();
	void loadLUT();
	void load();

	const cv::Mat_<cv::Vec3b>& getImg() const;
	const CubeLUT& getCube() const;
	const boost::program_options::variables_map& getVm() const;
};
