#pragma once
#include <FileIO/CubeLUT.hpp>
#include <opencv2/core.hpp>
#include <TaskDispatcher/InputParams.h>

class FileIO {
	cv::Mat3b img;
	std::string inputPath;
	std::string outputPath;
	std::string lutPath;
	bool forceOverwrite = false;

  public:
	explicit FileIO(const InputParams& params);
	virtual ~FileIO() = default;

	void setImg(cv::Mat newImage);

	bool loadImg();
	bool loadLut();
	bool saveImg(cv::Mat newImg) const;

	[[nodiscard]] cv::Mat3b& getImg();
	[[nodiscard]] cv::Mat1b& getAlpha();
	[[nodiscard]] const CubeLUT& getCube() const;

protected:
	std::unique_ptr<CubeLUT> cube;
	cv::Mat1b alphaChannel; // Used only for RGBA input images

	virtual cv::Mat readImage(const std::string& inputPath) const;
	virtual bool writeImage(const std::string &outputPath, cv::Mat newImg) const;
	virtual bool fileExists(std::error_code& ec) const;
};
