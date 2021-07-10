#include "initHandler.hpp"

InitHandler::InitHandler(const int aCnt, char* aVal[]) : arg_count(aCnt), args(aVal), loader(Loader())
{
}

int InitHandler::start()
{
	using namespace boost::program_options;
	variables_map vm;
	try
	{
		vm = parseInputArgs(arg_count, args);
	}
	catch (const error& ex)
	{
		std::cerr << ex.what() << '\n';
		return -1;
	}

	loader.setArgs(vm);

	try
	{
		loader.load();
	}
	catch (const std::exception& ex)
	{
		std::cerr << ex.what() << '\n';
		return -1;
	}


	Processor processor(loader);
	processor.perform();

	return 0;
}

boost::program_options::variables_map InitHandler::parseInputArgs(const int argc, char** argv) const
{
	using namespace boost::program_options;
	try
	{
		options_description desc{ "Options" };
		desc.add_options()
			("help,h", "Help screen")
			("input,i", value<std::string>(), "Input file path")
			("lut,l", value<std::string>(), "LUT file path")
			("output,o", value<std::string>()->default_value("out.png"), "Output file path")
			("strength,s", value<float>()->default_value(1.0f), "Strength of the effect")
			("trilinear,t", "Trilinear interpolation of 3D LUT")
			("nearest_value,n", "No interpolation of 3D LUT");
		variables_map vm;
		store(parse_command_line(argc, argv, desc), vm);
		if (vm.count("help"))
		{
			std::cout << "-- HELP --\n" << desc;
			throw error("");
		}
		if (!vm.count("input") || !vm.count("lut") || !vm.count("output"))
		{
			throw error("No input/output/LUT specified");
		}
		return vm;
	}
	catch (const error& ex)
	{
		throw;
	}
}
