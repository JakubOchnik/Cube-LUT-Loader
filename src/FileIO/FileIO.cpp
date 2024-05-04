#include <FileIO/FileIO.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fmt/format.h>

FileIO::FileIO(InputParams inputParams) : params(inputParams), cube(std::make_unique<CubeLUT>()) {}

bool FileIO::loadImg()
{
	std::cout << "[INFO] Importing image...\n";
	const auto inputPath = params.getInputImgPath();
	const auto sourceImg = readImage(inputPath);
	if (sourceImg.empty())
	{
		std::cerr << fmt::format("[ERROR] Could not open input image file: {}\n", inputPath);
		return false;
	}

	if (params.getOutputImageHeight() || params.getOutputImageWidth()) {
		unsigned int width = params.getOutputImageWidth() ? params.getOutputImageWidth() : sourceImg.size().width;
		unsigned int height = params.getOutputImageHeight() ? params.getOutputImageHeight() : sourceImg.size().height;

		std::cout << fmt::format("[INFO] Scaling image to {}x{}\n", width, height);
		resizeImage(sourceImg, img, width, height);
	} else {
		img = sourceImg;
	}

	return true;
}

cv::Mat FileIO::readImage(const std::string &inputPath)
{
	return cv::imread(inputPath);
}

void FileIO::resizeImage(cv::Mat inputImg, cv::Mat outputImg, unsigned int width, unsigned int height, int interpolationMode)
{
	cv::resize(inputImg, img, cv::Size(width, height), 0, 0, interpolationMode);
}

bool FileIO::loadLut()
{
	bool success = true;
	std::cout << "[INFO] Importing LUT...\n";
	const auto& lutPath = params.getInputLutPath();
	std::ifstream infile(lutPath);
	if (!infile.good())
	{
		std::cerr << fmt::format("[ERROR] Could not open input LUT file: {}\n", lutPath);
		return false;
	}
	std::cout << "[INFO] Parsing LUT...\n";
	try {
		cube->loadCubeFile(infile);
	} catch (const std::runtime_error& ex) {
		std::cerr << fmt::format("[ERROR] {}\n", ex.what());
		success = false;
	}
	infile.close();
	return success;
}

bool FileIO::load()
{
	return loadImg() && loadLut();
}

const cv::Mat_<cv::Vec3b>& FileIO::getImg() const
{
	return this->img;
}

const CubeLUT& FileIO::getCube() const
{
	return *cube;
}

const InputParams& FileIO::getInputParams() const
{
	return params;
}

uint FileIO::getThreads() const
{
	return params.getThreads();
}
