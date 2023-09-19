#pragma once
#include <DataLoader/CubeLUT.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>

class DataLoader
{
	cv::Mat_<cv::Vec3b>					  img;
	CubeLUT								  cube;
	boost::program_options::variables_map vm;

public:
	DataLoader(boost::program_options::variables_map varMap);
	void loadImg();
	void loadLut();
	void load();

	[[nodiscard]] const cv::Mat_<cv::Vec3b>&				   getImg() const;
	[[nodiscard]] const CubeLUT&							   getCube() const;
	[[nodiscard]] const boost::program_options::variables_map& getVm() const;
	[[nodiscard]] uint getThreads() const;
};
