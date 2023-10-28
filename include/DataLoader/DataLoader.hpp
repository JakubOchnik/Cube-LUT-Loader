#pragma once
#include <DataLoader/CubeLUT.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <TaskDispatcher/InputParams.h>

class DataLoader
{
	cv::Mat_<cv::Vec3b> img;
	CubeLUT cube;
	InputParams params;

public:
	explicit DataLoader(InputParams inputParams);
	bool loadImg();
	bool loadLut();
	bool load();

	[[nodiscard]] const cv::Mat_<cv::Vec3b>& getImg() const;
	[[nodiscard]] const CubeLUT& getCube() const;
	[[nodiscard]] const InputParams& getInputParams() const;
	[[nodiscard]] uint getThreads() const;
};
