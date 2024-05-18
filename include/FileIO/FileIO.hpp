#pragma once
#include <FileIO/CubeLUT.hpp>
#include <boost/program_options.hpp>
#include <opencv2/core.hpp>
#include <TaskDispatcher/InputParams.h>

class FileIO {
	cv::Mat_<cv::Vec3b> img;
	std::string inputPath;
	std::string outputPath;
	std::string lutPath;
	bool forceOverwrite = false;

  public:
	explicit FileIO(const InputParams& params);
	virtual ~FileIO() = default;

	void setImg(cv::Mat newImage);
	bool load();

	bool saveImg(cv::Mat newImg) const;

	[[nodiscard]] const cv::Mat_<cv::Vec3b>& getImg() const;
	[[nodiscard]] const CubeLUT& getCube() const;

protected:
	std::unique_ptr<CubeLUT> cube;
	bool loadImg();
	bool loadLut();

	virtual cv::Mat readImage(const std::string& inputPath) const;
	virtual bool writeImage(const std::string &outputPath, cv::Mat newImg) const;
	virtual bool fileExists(std::error_code& ec) const;
};
