#include <DataLoader/DataLoader.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <boost/format.hpp>

DataLoader::DataLoader(InputParams inputParams) : params(inputParams) {}

bool DataLoader::loadImg()
{
	std::cout << "[INFO] Importing image...\n";
	const auto& inputPath = params.getInputImgPath();
	img = cv::imread(inputPath);

	if (img.empty())
	{
		std::cerr << boost::format("[ERROR] Could not open input image file: %1%\n") % inputPath;
		return false;
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
