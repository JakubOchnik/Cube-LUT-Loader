#include <DataLoader/DataLoader.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <boost/format.hpp>

DataLoader::DataLoader(const boost::program_options::variables_map varMap) : vm(varMap) {}

bool DataLoader::loadImg()
{
	std::cout << "[INFO] Importing image...\n";
	img = cv::imread(vm["input"].as<std::string>());

	if (img.empty())
	{
		std::cerr << boost::format("[ERROR] Could not open input image file: %1%\n") % vm["input"].as<std::string>();
		return false;
	}
	return true;
}

bool DataLoader::loadLut()
{
	bool success = true;
	std::cout << "[INFO] Importing LUT...\n";
	std::ifstream infile(vm["lut"].as<std::string>());
	if (!infile.good())
	{
		std::cerr << boost::format("[ERROR] Could not open input LUT file: %1%\n") % vm["lut"].as<std::string>();
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

const boost::program_options::variables_map& DataLoader::getVm() const
{
	return this->vm;
}

uint DataLoader::getThreads() const
{
	return vm["threads"].as<uint>();
}
