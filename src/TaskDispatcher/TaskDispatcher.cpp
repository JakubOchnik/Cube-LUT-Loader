#ifdef BUILD_CUDA
#include <ImageProcessing/GPUImageProcess/GPUprocessor.hpp>
#endif
#include <ImageProcessing/CPUImageProcess/CPUProcessor.hpp>
#include <TaskDispatcher/TaskDispatcher.hpp>
#include <iostream>
#include <thread>
#include <iostream>

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
	InputParams parameters;
	try
	{
		std::string helpText;
		parameters = parseInputArgs(helpText);
		if (parameters.getShowHelp()) {
			std::cout << "--HELP--\n" << helpText;
			return SUCCESS_EXIT;
		}
	}
	catch (const std::exception &ex)
	{
		std::cerr << "[ERROR] " << ex.what() << '\n';
		return FAIL_EXIT;
	}
	FileIO fileIO{ parameters };

	bool loadSuccessful = fileIO.load();
	if (!loadSuccessful) {
		return FAIL_EXIT;
	}

	if (parameters.getProcessingMode() == ProcessingMode::GPU)
	{
#ifdef BUILD_CUDA
		std::cout << "[INFO] GPU acceleration enabled\n";
		GpuProcessor processor(fileIO);
		try
		{
			processor.execute();
		}
		catch (const std::exception &e)
		{
			std::cerr << "[ERROR] " << e.what() << '\n';
		}
#else
		std::cerr << "[ERROR] GPU acceleration is unsupported in this build\n";
#endif
	}
	else
	{
		std::cout << "[INFO] Using " << parameters.getThreads() << " CPU thread(s)\n";
		CPUProcessor processor(fileIO);
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

InputParams TaskDispatcher::parseInputArgs(std::string& helpText) const
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
	("threads,j", boost::program_options::value<unsigned int>()->default_value(std::thread::hardware_concurrency()),"Number of threads [= Number of physical threads]")
	("gpu", "Use GPU acceleration")
	("width", boost::program_options::value<int>(), "Output image width")
	("height", boost::program_options::value<int>(), "Output image height");

	boost::program_options::variables_map vm;
	store(parse_command_line(argCount, args, desc), vm);

	if (vm.count("help"))
	{
		std::stringstream ss;
		ss << desc;
		helpText = ss.str();
		InputParams params;
		params.setShowHelp(true);
		return params;
	}

	if (!vm.count("input") || !vm.count("lut"))
	{
		throw boost::program_options::error("No input image/LUT specified!");
	}

	if (vm.count("trilinear") && vm.count("nearest_value")) {
		std::cout << "[WARNING] Ambiguous input: multiple interpolation methods specified. Using trilinear.\n";
	}

	if ((vm.count("width") || vm.count("height")) && !(vm.count("width") && vm.count("height"))) {
		std::cout << "[WARNING] Not all output image dimensions have been specified.\n";
	}

	return InputParams{ std::move(vm) };
}
