#ifdef BUILD_CUDA
#include <ImageProcessing/GPU/GPUModeExecutor.hpp>
#include <ImageProcessing/GPU/Utils/CudaUtils.hpp>
#endif
#include <ImageProcessing/CPU/CPUModeExecutor.hpp>
#include <TaskDispatcher/TaskDispatcher.hpp>
#include <iostream>
#include <thread>
#include <args.hxx>

enum {
	FAIL_EXIT = -1,
	SUCCESS_EXIT
};

TaskDispatcher::TaskDispatcher(const int aCnt, char *aVal[])
	: argCount(aCnt), argv(aVal)
{
}

int TaskDispatcher::start()
{
	InputParams parameters;
	try
	{
		parameters = parseInputArgs();
	}
	catch (const args::Help&) {
		return SUCCESS_EXIT;
	}
	catch (const std::runtime_error&) {
		return FAIL_EXIT;
	}
	FileIO fileIO{ parameters };

	bool loadSuccessful{false};
	try {
		loadSuccessful = fileIO.loadImg() && fileIO.loadLut();
	} catch (const std::exception& ex) {
		std::cerr << fmt::format("[ERROR] Fatal exception: {}", ex.what());
		return FAIL_EXIT;
	}

	if (!loadSuccessful) {
		return FAIL_EXIT;
	}

	cv::Mat finalImage;
	if (parameters.getProcessingMode() == ProcessingMode::GPU) 
	{
#ifdef BUILD_CUDA
		if (!CudaUtils::isCudaAvailable()) {
			return FAIL_EXIT;
		}
		std::cout << "[INFO] GPU acceleration enabled\n";
		GPUModeExecutor processor(fileIO);
		try
		{
			finalImage = processor.execute(parameters.getEffectIntensity(),
										   {parameters.getOutputImageWidth(), parameters.getOutputImageHeight()},
										   parameters.getInterpolationMethod());
		} catch (const std::exception& e) {
			std::cerr << "[ERROR] " << e.what() << '\n';
			return FAIL_EXIT;
		}

		// Currently needs to be done here - unified memory is freed in the destructor of GPUModeExecutor
		// TODO: Separate output mat lifetime from GPUModeExecutor and don't use the default cv::Mat deallocator (bug-prone and unsafe)
		if (!fileIO.saveImg(finalImage)) {
			return FAIL_EXIT;
		}
#else
		std::cerr << "[ERROR] GPU acceleration is unsupported in this build\n";
		return FAIL_EXIT;
#endif
	} else {
		std::cout << "[INFO] Using " << parameters.getThreads() << " CPU thread(s)\n";
		CPUModeExecutor processor(fileIO, parameters.getThreads());
		try
		{
			finalImage = processor.execute(parameters.getEffectIntensity(),
										   {parameters.getOutputImageWidth(), parameters.getOutputImageHeight()},
										   parameters.getInterpolationMethod());
		} catch (const std::exception& e) {
			std::cerr << e.what() << '\n';
			return FAIL_EXIT;
		}

		if (!fileIO.saveImg(finalImage)) {
			return FAIL_EXIT;
		}
	}

	return SUCCESS_EXIT;
}

namespace {
	float clipStrength(float strength) {
		if (strength <= .0f) {
			std::cout << fmt::format("[WARNING] Incorrect strength ({}) - clipping to 0 %.\n", strength);
			return .0f;
		}
		if (strength > 100.0f) {
			std::cout << fmt::format("[WARNING] Incorrect strength ({}) - clipping to 100 %.\n", strength);
			return 100.0f;
		}

		return strength;
	}

	int clipDimension(int imageDimension, std::string_view name) {
		if (imageDimension <= 0) {
			std::cout << fmt::format("[WARNING] Incorrect image {} ({}) - ignoring.\n", name, imageDimension);
			return 0;
		}

		return imageDimension;
	}

	const std::unordered_map<std::string, InterpolationMethod> interpolationMethodMapping {
		{"trilinear", InterpolationMethod::Trilinear},
		{"nearest-value", InterpolationMethod::NearestValue}
	};

	const std::unordered_map<std::string, ProcessingMode> processingModeMapping {
		{"cpu", ProcessingMode::CPU},
		{"gpu", ProcessingMode::GPU}
	};

	struct ToLowerReader {
	bool operator()(const std::string &name, const std::string &value, std::string &destination)
	{
		destination = value;
		std::transform(destination.begin(), destination.end(), destination.begin(), ::tolower);
		return true;
	}
};
}

InputParams TaskDispatcher::parseInputArgs() const
{
	args::ArgumentParser parser("Cube LUT Loader");
	args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
	parser.Prog("Cube LUT Loader");
	parser.SetArgumentSeparations(false, true, true, true);
	args::ValueFlag<std::string> input(parser, "input", "Input file path", {'i', "input"}, args::Options::Required);
	args::ValueFlag<std::string> lut(parser, "lut", "LUT file path", {'l', "lut"}, args::Options::Required);
	args::ValueFlag<std::string> output(parser, "output", "Output file path", {'o', "output"});
	args::ValueFlag<float> strength(parser, "intensity", "Intensity of the applied LUT (0-100)", {'s', "strength"}, 100.0f);
	args::MapFlag<std::string, InterpolationMethod, ToLowerReader> method(parser, "method", "Interpolation method (allowed values: 'trilinear', 'nearest-value')", {'m', "method"},
														   interpolationMethodMapping, InterpolationMethod::Trilinear, args::Options::Single);
	args::Flag forceOverwrite(parser, "force", "Force overwrite file", {'f', "force"});
	const unsigned int defaultNumberOfThreads = std::thread::hardware_concurrency();
	args::ValueFlag<unsigned int> threads(parser, "threads", "Number of threads", {'j', "threads"}, defaultNumberOfThreads);
	args::MapFlag<std::string, ProcessingMode, ToLowerReader> processingMode(parser, "processor", "Processing mode (allowed values: 'cpu', 'gpu')", {'p', "processor"},
														processingModeMapping, ProcessingMode::CPU, args::Options::Single);
	args::ValueFlag<int> width(parser, "width", "Output image width", {"width"});
	args::ValueFlag<int> height(parser, "height", "Output image height", {"height"});

	try {
		parser.ParseCLI(argCount, argv);
	} catch(const args::Help& helpEx) {
		std::cout << parser;
		throw helpEx;
	} catch (const args::Error& ex) {
		std::cerr << fmt::format("[ERROR] {}\n", ex.what());
		throw ex;
	}

	if ((width || height) && !(width && height)) {
		std::cout << "[WARNING] Not all output image dimensions have been specified.\n";
	}

	return InputParams {
		*processingMode,
		*threads,
		*method,
		*input,
		*output,
		forceOverwrite,
		*lut,
		clipStrength(*strength),
		width ? clipDimension(*width, "width") : 0,
		height ? clipDimension(*height, "height") : 0
	};
}
