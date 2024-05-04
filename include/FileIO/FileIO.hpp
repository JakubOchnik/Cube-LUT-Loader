#pragma once
#include <FileIO/CubeLUT.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <TaskDispatcher/InputParams.h>

class FileIO
{
	cv::Mat_<cv::Vec3b> img;
	std::string inputPath;
	std::string outputPath;
	std::string lutPath;

public:
	explicit FileIO(const std::string& inputPath, const std::string& outputPath, const std::string& lutPath);
	explicit FileIO(const InputParams& params);
	virtual ~FileIO() = default;

	bool loadImg();
	void setImg(cv::Mat newImage);
	bool loadLut();
	bool load();

	bool saveImg(cv::Mat newImg) const;

	[[nodiscard]] const cv::Mat_<cv::Vec3b>& getImg() const;
	[[nodiscard]] const CubeLUT& getCube() const;

protected:
	std::unique_ptr<CubeLUT> cube;

	virtual cv::Mat readImage(const std::string& inputPath);
};
