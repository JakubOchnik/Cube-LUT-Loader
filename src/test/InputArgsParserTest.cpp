#include <gtest/gtest.h>
#include <TaskDispatcher/TaskDispatcher.hpp>
#include <string.h>
#include <args.hxx>

using namespace ::testing;

class Array2D {
public:
	Array2D() = default;

	Array2D(const std::vector<std::string>& args) {
        initializeArray(args);
    }

	void reset(const std::vector<std::string>& args) {
		free2Darray();
		initializeArray(args);
	}

	~Array2D() {
        free2Darray();
    }

	char** arguments = nullptr;
	size_t argCount = 0;

private:
	void initializeArray(const std::vector<std::string>& args) {
		argCount = args.size() + 1;
		arguments = new char*[argCount];
		for (int i{}; i < argCount - 1; ++i) {
			const auto& argument = args[i];
			arguments[i] = new char[argument.size() + 1];
			strncpy(arguments[i], argument.c_str(), argument.size() + 1);
		}
		// Final argument has to be nullptr
		arguments[argCount - 1] = nullptr;
	}

	void free2Darray() {
		if (!arguments) {
			return;
		}

		if (argCount <= 0) {
			return;
		}

		for (int i{}; i < argCount - 1; ++i) {
			delete arguments[i];
		}
		delete arguments;
		arguments = nullptr;
		argCount = 0;
	}
};

class InputArgsParserTest : public ::testing::Test {};

TEST_F(InputArgsParserTest, testHelp)
{
    const std::vector<std::string> sourceArgs{"program", "-h"};
    Array2D argsArr(sourceArgs);
    const auto rawArgs = argsArr.arguments;

    TaskDispatcher dispatcher(sourceArgs.size(), rawArgs);
    EXPECT_THROW(dispatcher.parseInputArgs(), args::Help);
}

TEST_F(InputArgsParserTest, emptyArgs)
{
    const std::vector<std::string> sourceArgs{"program"};
    Array2D argsArr(sourceArgs);
    const auto rawArgs = argsArr.arguments;

    TaskDispatcher dispatcher(sourceArgs.size(), rawArgs);
    EXPECT_THROW(dispatcher.parseInputArgs(), args::Error);
}

TEST_F(InputArgsParserTest, strength)
{
    const std::vector<std::string> baseArray{"program", "-i", "abc.png", "-l", "test.cube"};
    std::vector<std::string> arrayWithIntensity = baseArray;

    arrayWithIntensity.push_back("--intensity=42");
    Array2D argsArr(arrayWithIntensity);
    auto rawArgumentsArray = argsArr.arguments;
    TaskDispatcher dispatcher(arrayWithIntensity.size(), rawArgumentsArray);
    auto params = dispatcher.parseInputArgs();
    EXPECT_EQ(params.getEffectIntensity(), .42f);
    arrayWithIntensity.pop_back();

    arrayWithIntensity.push_back("--intensity=101");
    argsArr.reset(arrayWithIntensity);
    rawArgumentsArray = argsArr.arguments;
    dispatcher = TaskDispatcher(arrayWithIntensity.size(), rawArgumentsArray);
    params = dispatcher.parseInputArgs();
    EXPECT_EQ(params.getEffectIntensity(), 1.0f);
    arrayWithIntensity.pop_back();

    arrayWithIntensity.push_back("--intensity=-2");
    argsArr.reset(arrayWithIntensity);
    rawArgumentsArray = argsArr.arguments;
    dispatcher = TaskDispatcher(arrayWithIntensity.size(), rawArgumentsArray);
    params = dispatcher.parseInputArgs();
    EXPECT_EQ(params.getEffectIntensity(), .0f);
}

struct IncorrectDimensionsTest : public InputArgsParserTest, public ::testing::WithParamInterface<std::tuple<std::vector<std::string>, int, int, bool>> {};

TEST_P(IncorrectDimensionsTest, incorrectDimensions) {
    const auto&[additionalArgs, expectedWidth, expectedHeight, shouldFail] = GetParam();
    std::vector<std::string> arguments{"program", "-i", "abc.png", "-l", "test.cube"};
    arguments.insert(arguments.end(), additionalArgs.begin(), additionalArgs.end());
    Array2D argsArr(arguments);
    auto rawArgumentsArray = argsArr.arguments;
    TaskDispatcher dispatcher(arguments.size(), rawArgumentsArray);
    if (shouldFail) {
        EXPECT_THROW(dispatcher.parseInputArgs(), args::Error);
    } else {
        auto params = dispatcher.parseInputArgs();
        EXPECT_EQ(params.getOutputImageWidth(), expectedWidth);
        EXPECT_EQ(params.getOutputImageHeight(), expectedHeight);
    }
}

INSTANTIATE_TEST_SUITE_P(
        InputArgsParserTest,
        IncorrectDimensionsTest,
        ::testing::Values(
            std::make_tuple(std::vector<std::string>{"--width=-10"}, 0, 0, false),
            std::make_tuple(std::vector<std::string>{"--height=0.5"}, 0, 0, true),
            std::make_tuple(std::vector<std::string>{"--height=10", "--width=-2"}, 0, 10, false),
            std::make_tuple(std::vector<std::string>{"--height=-2", "--width=10"}, 10, 0, false),
            std::make_tuple(std::vector<std::string>{"--height=0", "--width=0"}, 0, 0, false)) // Default
);


struct IncorrectInputTest : public InputArgsParserTest, public ::testing::WithParamInterface<std::vector<std::string>> {};

TEST_P(IncorrectInputTest, incorrectParam) {
    std::vector<std::string> sourceArgs{"program"};
    const auto& testArgs = GetParam();
    sourceArgs.insert(sourceArgs.end(), testArgs.begin(), testArgs.end());
    Array2D argsArr(sourceArgs);
    const auto arguments = argsArr.arguments;

    TaskDispatcher dispatcher(sourceArgs.size(), arguments);
    EXPECT_THROW(dispatcher.parseInputArgs(), args::Error);
}

INSTANTIATE_TEST_SUITE_P(
        InputArgsParserTest,
        IncorrectInputTest,
        ::testing::Values(
            std::vector<std::string>{},
            std::vector<std::string>{"-l", "abcd.cube"},
            std::vector<std::string>{"-i", "abcd.png"},
            std::vector<std::string>{"--interpolation", "trilinear"},
            std::vector<std::string>{"-l", "--interpolation", "-i"},
            std::vector<std::string>{"-l", "abcd.cube", "-i", "abcd.png", "--abc"},
            std::vector<std::string>{"-l", "abcd.cube", "-i", "abcd.png", "--intensity"},
            std::vector<std::string>{"-l", "abcd.cube", "-i", "abcd.png", "--intensity="},
            std::vector<std::string>{"-l", "abcd.cube", "-i", "abcd.png", "--intensity", "-t"}
        )
);

struct ModesTest : public InputArgsParserTest, public ::testing::WithParamInterface<std::tuple<std::vector<std::string>, ProcessingMode, InterpolationMethod, bool>> {};

TEST_P(ModesTest, processingModes) {
   std::vector<std::string> sourceArgs{"program", "-l", "abcd.cube", "-i", "abcd.png"};
    const auto& [testArgs, expectedMode, expectedMethod, shouldFail] = GetParam();
    sourceArgs.insert(sourceArgs.end(), testArgs.begin(), testArgs.end());
    Array2D argsArr(sourceArgs);
    const auto arguments = argsArr.arguments;

    TaskDispatcher dispatcher(sourceArgs.size(), arguments);
    if (shouldFail) {
        EXPECT_THROW(dispatcher.parseInputArgs(), args::Error);
    } else {
        const auto params = dispatcher.parseInputArgs();
        EXPECT_EQ(params.getProcessingMode(), expectedMode);
        EXPECT_EQ(params.getInterpolationMethod(), expectedMethod);
    }
}

INSTANTIATE_TEST_SUITE_P(
        InputArgsParserTest,
        ModesTest,
        ::testing::Values(
            std::make_tuple(std::vector<std::string>{}, ProcessingMode::CPU, InterpolationMethod::Trilinear, false),
            std::make_tuple(std::vector<std::string>{"--interpolation", "trilinear"}, ProcessingMode::CPU, InterpolationMethod::Trilinear, false),
            std::make_tuple(std::vector<std::string>{"--interpolation", "Trilinear"}, ProcessingMode::CPU, InterpolationMethod::Trilinear, false),
            std::make_tuple(std::vector<std::string>{"--interpolation", "nearest-value"}, ProcessingMode::CPU, InterpolationMethod::NearestValue, false),
            std::make_tuple(std::vector<std::string>{"--interpolation", "Nearest-value"}, ProcessingMode::CPU, InterpolationMethod::NearestValue, false),
            std::make_tuple(std::vector<std::string>{"--interpolation", "blabla"}, ProcessingMode::CPU, InterpolationMethod::Trilinear, true),
            std::make_tuple(std::vector<std::string>{"--interpolation", "trilinear", "--interpolation", "nearest-value"}, ProcessingMode::CPU, InterpolationMethod::Trilinear, true),
            std::make_tuple(std::vector<std::string>{"-p", "cpu"}, ProcessingMode::CPU, InterpolationMethod::Trilinear, false),
            std::make_tuple(std::vector<std::string>{"-p", "CPU"}, ProcessingMode::CPU, InterpolationMethod::Trilinear, false),
            std::make_tuple(std::vector<std::string>{"-p", "gpu"}, ProcessingMode::GPU, InterpolationMethod::Trilinear, false),
            std::make_tuple(std::vector<std::string>{"-p", "GPU"}, ProcessingMode::GPU, InterpolationMethod::Trilinear, false),
            std::make_tuple(std::vector<std::string>{"-p", "blabla"}, ProcessingMode::CPU, InterpolationMethod::Trilinear, true),
            std::make_tuple(std::vector<std::string>{"-p", "cpu", "-p", "gpu"}, ProcessingMode::CPU, InterpolationMethod::Trilinear, true)
        )
);
