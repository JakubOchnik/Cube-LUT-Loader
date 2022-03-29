#include <DataLoader/dataLoader.hpp>

DataLoader::DataLoader(const boost::program_options::variables_map varMap) : vm(varMap)
{
}

void DataLoader::setArgs(const boost::program_options::variables_map varMap)
{
	this->vm = varMap;
}

void DataLoader::loadImg()
{
	std::cout << "Importing image...\n";
	img = cv::imread(vm["input"].as<std::string>());

	if (img.empty())
	{
		const std::string msg{"Could not open input file: " + vm["input"].as<std::string>()};
		throw std::runtime_error(msg.c_str());
	}
}

void DataLoader::loadLut()
{
	std::cout << "Importing LUT...\n";
	std::ifstream infile(vm["lut"].as<std::string>());
	if (!infile.good())
	{
		const std::string msg{"Could not open input LUT file: " + vm["lut"].as<std::string>()};
		throw std::runtime_error(msg.c_str());
	}
	std::cout << "Parsing LUT...\n";
	const int ret = cube.LoadCubeFile(infile);
	infile.close();
	if (ret != 0)
	{
		const std::string msg{"Could not parse the cube info in the input file. Return code = " + std::to_string(ret)};
		throw std::runtime_error(msg.c_str());
	}
}

void DataLoader::load()
{
	loadImg();
	loadLut();
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
