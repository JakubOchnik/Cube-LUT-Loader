#include <Loader/Loader.hpp>

Loader::Loader(const boost::program_options::variables_map vm) : vm(vm)
{
}

Loader::Loader()
{
}

void Loader::setArgs(const boost::program_options::variables_map vm)
{
	this->vm = vm;
}

void Loader::loadImg()
{
	std::cout << "Importing image...\n";
	img = cv::imread(vm["input"].as<std::string>());
}

void Loader::loadLUT()
{
	std::cout << "Importing LUT...\n";
	ifstream infile(vm["lut"].as<std::string>());
	if (!infile.good())
	{
		std::string msg = "Could not open input LUT file: " + vm["lut"].as<std::string>();
		throw std::exception(msg.c_str());
	}
	std::cout << "Parsing LUT...\n";
	const int ret = cube.LoadCubeFile(infile);
	infile.close();
	if (ret != 0)
	{
		std::string msg = "Could not parse the cube info in the input file. Return code = " + ret;
		throw std::exception(msg.c_str());
	}
}

void Loader::load()
{
	try
	{
		loadImg();
		loadLUT();
	}
	catch (const std::exception& ex)
	{
		throw;
	}
}

const cv::Mat_<cv::Vec3b>& Loader::getImg() const
{
	return this->img;
}

const CubeLUT& Loader::getCube() const
{
	return this->cube;
}

const boost::program_options::variables_map& Loader::getVm() const
{
	return this->vm;
}

const uint Loader::getThreads() const
{
	return vm["threads"].as<uint>();
}
