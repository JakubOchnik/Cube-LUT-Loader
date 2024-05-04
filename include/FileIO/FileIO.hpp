#pragma once
#include <FileIO/CubeLUT.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <TaskDispatcher/InputParams.h>

class FileIO
{
	cv::Mat_<cv::Vec3b> img;
	InputParams params;

public:
	explicit FileIO(InputParams inputParams);
	bool loadImg();
	bool loadLut();
	bool load();

	[[nodiscard]] const cv::Mat_<cv::Vec3b>& getImg() const;
	[[nodiscard]] const CubeLUT& getCube() const;
	[[nodiscard]] const InputParams& getInputParams() const;
	[[nodiscard]] uint getThreads() const;

protected:
	std::unique_ptr<CubeLUT> cube;

	virtual cv::Mat readImage(const std::string& inputPath);
	virtual void resizeImage(cv::Mat inputImg, cv::Mat outputImg, unsigned int width, unsigned int height, int interpolationMode = 2);
};
