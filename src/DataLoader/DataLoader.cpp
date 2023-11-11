#include <DataLoader/DataLoader.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/format.hpp>

DataLoader::DataLoader(InputParams inputParams) : params(inputParams) {}

bool DataLoader::loadImg()
{
	std::cout << "[INFO] Importing image...\n";
	const auto inputPath = params.getInputImgPath();
	const auto sourceImg = cv::imread(inputPath);
	if (sourceImg.empty())
	{
		std::cerr << boost::format("[ERROR] Could not open input image file: %1%\n") % inputPath;
		return false;
	}

	if (params.getOutputImageHeight() || params.getOutputImageWidth()) {
		unsigned int width = params.getOutputImageWidth() ? params.getOutputImageWidth() : sourceImg.size().width;
		unsigned int height = params.getOutputImageHeight() ? params.getOutputImageHeight() : sourceImg.size().height;
		cv::Size newSize(width, height);

		std::cout << boost::format("[INFO] Scaling image to %1%x%2%\n") % width % height;
		cv::resize(sourceImg, img, newSize, 0, 0, cv::InterpolationFlags::INTER_CUBIC);
	} else {
		img = sourceImg;
	}

	return true;
}

bool DataLoader::loadLut()
{
	bool success = true;
	std::cout << "[INFO] Importing LUT...\n";
	const auto& lutPath = params.getInputLutPath();
	std::ifstream infile(lutPath);
	if (!infile.good())
	{
		std::cerr << boost::format("[ERROR] Could not open input LUT file: %1%\n") % lutPath;
		return false;
	}
	std::cout << "[INFO] Parsing LUT...\n";
	try {
		cube.loadCubeFile(infile);
	} catch (const std::runtime_error& ex) {
		std::cerr << boost::format("[ERROR] %1%\n") % ex.what();
		success = false;
	}
	infile.close();
	return success;
}

bool DataLoader::load()
{
	return loadImg() && loadLut();
}

const cv::Mat_<cv::Vec3b>& DataLoader::getImg() const
{
	return this->img;
}

const CubeLUT& DataLoader::getCube() const
{
	return this->cube;
}

const InputParams& DataLoader::getInputParams() const
{
	return params;
}

uint DataLoader::getThreads() const
{
	return params.getThreads();
}
