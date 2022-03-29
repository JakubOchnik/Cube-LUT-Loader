#include <TaskDispatcher/TaskDispatcher.hpp>

TaskDispatcher::TaskDispatcher(const int aCnt, char* aVal[]) : argCount(aCnt), args(aVal), loader(DataLoader())
{
}

int TaskDispatcher::start()
{
	boost::program_options::variables_map vm;
	try
	{
		vm = parseInputArgs(argCount, args);
	}
	catch (const boost::program_options::error& ex)
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

	if (loader.getVm().count("gpu"))
	{
		// TODO: Add verification (driver & compute capability),
		// possibly display some device info.
		cout << "GPU acceleration enabled\n";
		GpuProcessor processor(loader);
		try
		{
			processor.execute();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
		}
	}
	else
	{
		cout << "Using " << loader.getThreads() << " CPU thread(s)\n";
		Processor processor(loader);
		try
		{
			processor.execute();
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << '\n';
		}
	}
	return 0;
}

boost::program_options::variables_map TaskDispatcher::parseInputArgs(const int argc, char** argv) const
{
	boost::program_options::options_description desc{"Options"};
	desc.add_options()
		("help,h", "Help screen")
		("input,i", boost::program_options::value<std::string>(), "Input file path")
		("lut,l", boost::program_options::value<std::string>(), "LUT file path")
		("output,o",
		 boost::program_options::value<std::string>()->default_value("out.png"),
		 "Output file path [= out.png]")
		("strength,s",
		 boost::program_options::value<float>()->default_value(1.0f),
		 "Strength of the effect [= 1.0]")
		("trilinear,t", "Trilinear interpolation of 3D LUT")
		("nearest_value,n", "No interpolation of 3D LUT")
		("threads,j",
		 boost::program_options::value<uint>()->default_value(std::thread::hardware_concurrency()),
		 "Number of threads [= Number of physical threads]")
		("gpu", "Use GPU acceleration")
		("test", "Performance test mode");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);

	if (vm.count("help"))
	{
		std::cout << "-- HELP --\n" << desc;
		// TODO: not a greatest solution ;)
		// TODO: Improvement: use external logger class for logging
		throw boost::program_options::error("");
	}
	if (!vm.count("input") || !vm.count("lut") || !vm.count("output"))
	{
		throw boost::program_options::error("No input/output/LUT specified!");
	}

	return vm;
}
