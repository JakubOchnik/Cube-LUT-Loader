#include "initHandler.hpp"

InitHandler::InitHandler(const int aCnt, char** aVal) :arg_count(aCnt), args(aVal), loader(Loader())
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
	catch (const std::invalid_argument& ex)
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
			("strength,s", value<float>()->default_value(1.0f), "Strength of the effect");
		variables_map vm;
		store(parse_command_line(argc, argv, desc), vm);
		if (vm.count("help"))
		{
			std::cout << "-- HELP --\n" << desc;
		}
		return vm;
	}
	catch (const error& ex)
	{
		std::cerr << ex.what() << '\n';
		throw std::invalid_argument("Invalid argument");
	}
}