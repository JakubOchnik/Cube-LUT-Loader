#ifdef BUILD_CUDA
#include <ImageProcessing/GPUImageProcess/GPUprocessor.hpp>
#endif
#include <ImageProcessing/CPUImageProcess/CPUProcessor.hpp>
#include <TaskDispatcher/TaskDispatcher.hpp>
#include <iostream>
#include <thread>
#include <iostream>

using namespace boost::program_options;

enum {
	FAIL_EXIT = -1,
	SUCCESS_EXIT
};

TaskDispatcher::TaskDispatcher(const int aCnt, char *aVal[])
	: argCount(aCnt), args(aVal)
{
}

int TaskDispatcher::start()
{
	variables_map vm;
	try
	{
		const auto parseOutput = parseInputArgs(argCount, args);
		if (std::holds_alternative<options_description>(parseOutput)) {
			std::cout << "--HELP--\n" << std::get<options_description>(parseOutput);
			return 0;
		}
		vm = std::move(std::get<variables_map>(parseOutput));
	}
	catch (const boost::program_options::error &ex)
	{
		std::cerr << ex.what() << '\n';
		return FAIL_EXIT;
	}
	DataLoader loader{vm};

	bool loadSuccessful = loader.load();
	if (!loadSuccessful) {
		return FAIL_EXIT;
	}

	if (loader.getVm().count("gpu"))
	{
#ifdef BUILD_CUDA
		std::cout << "[INFO] GPU acceleration enabled\n";
		GpuProcessor processor(loader);
		try
		{
			processor.execute();
		}
		catch (const std::exception &e)
		{
			std::cerr << e.what() << '\n';
		}
#else
		std::cerr << "[ERROR] GPU acceleration is unsupported in this build\n";
#endif
	}
	else
	{
		std::cout << "[INFO] Using " << loader.getThreads() << " CPU thread(s)\n";
		CPUProcessor processor(loader);
		try
		{
			processor.execute();
		}
		catch (const std::exception &e)
		{
			std::cerr << e.what() << '\n';
		}
	}
	return SUCCESS_EXIT;
}

std::variant<TaskDispatcher::VariablesMap, TaskDispatcher::OptionsDescription> TaskDispatcher::parseInputArgs(const int argc, char **argv) const
{
	boost::program_options::options_description desc{"Options"};
	desc.add_options()
	("help,h", "Help screen")
	("input,i", boost::program_options::value<std::string>(), "Input file path")
	("lut,l", boost::program_options::value<std::string>(), "LUT file path")
	("output,o", boost::program_options::value<std::string>()->default_value("out.png"), "Output file path [= out.png]")
	("strength,s", boost::program_options::value<float>()->default_value(1.0f), "Strength of the effect [= 1.0]")
	("trilinear,t", "Trilinear interpolation of 3D LUT")
	("nearest_value,n", "No interpolation of 3D LUT")
	("threads,j", boost::program_options::value<uint>()->default_value(std::thread::hardware_concurrency()),"Number of threads [= Number of physical threads]")
	("gpu", "Use GPU acceleration");

	boost::program_options::variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);

	if (vm.count("help"))
	{
		return desc;
	}
	if (!vm.count("input") || !vm.count("lut") || !vm.count("output"))
	{
		throw boost::program_options::error("No input/output/LUT specified!");
	}

	return vm;
}
