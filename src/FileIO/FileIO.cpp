#include <FileIO/FileIO.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fmt/format.h>

FileIO::FileIO(const std::string& inputPath, const std::string& outputPath, const std::string& lutPath)
	: inputPath(inputPath), outputPath(outputPath), lutPath(lutPath), cube(std::make_unique<CubeLUT>()) {}

FileIO::FileIO(const InputParams& params)
	: inputPath(params.getInputImgPath()), outputPath(params.getOutputImgPath()), lutPath(params.getInputLutPath()),
	  cube(std::make_unique<CubeLUT>()) {}

bool FileIO::loadImg() {
	std::cout << "[INFO] Importing image...\n";
	const auto sourceImg = readImage(inputPath);
	if (sourceImg.empty()) {
		std::cerr << fmt::format("[ERROR] Could not open input image file: {}\n", inputPath);
		return false;
	}
	img = sourceImg;
	return true;
}

void FileIO::setImg(cv::Mat newImage) {
	img = newImage;
}

cv::Mat FileIO::readImage(const std::string &inputPath)
{
	return cv::imread(inputPath);
}

bool FileIO::loadLut()
{
	bool success = true;
	std::cout << "[INFO] Importing LUT...\n";
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

bool FileIO::saveImg(cv::Mat newImg) const {
	std::cout << fmt::format("[INFO] Saving image to: {}\n", outputPath);
	try {
		cv::imwrite(outputPath, newImg);
	} catch (cv::Exception& ex) {
		std::cerr << fmt::format("[ERROR] {}\n", ex.what());
		return false;
	}
	return true;
}
