#include <FileIO/FileIO.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fmt/format.h>

FileIO::FileIO(const InputParams& params)
	: inputPath(params.getInputImgPath()), outputPath(params.getOutputImgPath()), lutPath(params.getInputLutPath()),
	  forceOverwrite(params.getForceOverwrite()), cube(std::make_unique<CubeLUT>()) {}

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

cv::Mat FileIO::readImage(const std::string& inputPath) const {
	return cv::imread(inputPath);
}

bool FileIO::writeImage(const std::string& outputPath, cv::Mat newImg) const {
	return cv::imwrite(outputPath, newImg);
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

bool FileIO::fileExists(std::error_code& ec) const {
	return std::filesystem::exists(outputPath, ec);
}

bool FileIO::saveImg(cv::Mat newImg) const {
	std::cout << fmt::format("[INFO] Saving image to: {}\n", outputPath);
	if (!forceOverwrite) {
		std::error_code ec;
		const auto alreadyExists = fileExists(ec);
		if (ec) {
			std::cerr << fmt::format("[ERROR] Failed to check if the file exists: {} {}\n", ec.value(), ec.message());
			return false;
		} else if (alreadyExists) {
			std::cerr << fmt::format("[ERROR] File {} already exists. Use -f to force overwrite.\n", outputPath);
			return false;
		}
	}

	bool success = false;
	try {
		success = writeImage(outputPath, newImg);
	} catch (cv::Exception& ex) {
		std::cerr << fmt::format("[ERROR] {}\n", ex.what());
		return false;
	}
	if (!success) {
		std::cerr << fmt::format("[ERROR] Failed to save the file: {}\n", outputPath);
		return false;
	}
	return true;
}
