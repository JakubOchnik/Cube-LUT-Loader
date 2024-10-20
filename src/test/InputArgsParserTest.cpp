#include <gtest/gtest.h>
#include <TaskDispatcher/TaskDispatcher.hpp>
#include <string.h>
#include <args.hxx>

using namespace ::testing;

class InputArgsParserTest : public ::testing::Test {
protected:
    char** arguments;
    size_t argCount = 0;

    char** initialize2DArray(const std::vector<std::string>& args) {
        argCount = args.size() + 1;
        arguments = new char*[argCount];
        for (int i{}; i < argCount - 1; ++i) {
            const auto& argument = args[i];
            arguments[i] = new char[argument.size() + 1];
            strncpy(arguments[i], argument.c_str(), argument.size() + 1);
        }
        // Final argument has to be nullptr
        arguments[argCount - 1] = nullptr;
        return arguments;
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
    }

    void TearDown() override {
        free2Darray();
    }
};

TEST_F(InputArgsParserTest, testHelp)
{
    const std::vector<std::string> sourceArgs{"program", "-h"};
    const auto arguments = initialize2DArray(sourceArgs);

    TaskDispatcher dispatcher(sourceArgs.size(), arguments);
    EXPECT_THROW(dispatcher.parseInputArgs(), args::Help);
}

TEST_F(InputArgsParserTest, emptyArgs)
{
    const std::vector<std::string> sourceArgs{"program"};
    const auto arguments = initialize2DArray(sourceArgs);

    TaskDispatcher dispatcher(sourceArgs.size(), arguments);
    EXPECT_THROW(dispatcher.parseInputArgs(), args::Error);
}

TEST_F(InputArgsParserTest, strength)
{
    const std::vector<std::string> baseArray{"program", "-i", "abc.png", "-l", "test.cube"};
    std::vector<std::string> arrayWithStrength = baseArray;

    arrayWithStrength.push_back("--strength=42");
    auto rawArgumentsArray = initialize2DArray(arrayWithStrength);
    TaskDispatcher dispatcher(arrayWithStrength.size(), rawArgumentsArray);
    auto params = dispatcher.parseInputArgs();
    EXPECT_EQ(params.getEffectStrength(), .42f);
    free2Darray();
    arrayWithStrength.pop_back();

    arrayWithStrength.push_back("--strength=101");
    rawArgumentsArray = initialize2DArray(arrayWithStrength);
    dispatcher = TaskDispatcher(arrayWithStrength.size(), rawArgumentsArray);
    params = dispatcher.parseInputArgs();
    EXPECT_EQ(params.getEffectStrength(), 1.0f);
    free2Darray();
    arrayWithStrength.pop_back();

    arrayWithStrength.push_back("--strength=-2");
    rawArgumentsArray = initialize2DArray(arrayWithStrength);
    dispatcher = TaskDispatcher(arrayWithStrength.size(), rawArgumentsArray);
    params = dispatcher.parseInputArgs();
    EXPECT_EQ(params.getEffectStrength(), .0f);
}

TEST_F(InputArgsParserTest, multipleInterpolationMethods)
{
    const std::vector<std::string> sourceArgs{"program", "-i", "abc.png", "-l", "test.cube", "-t", "-n"};
    const auto arguments = initialize2DArray(sourceArgs);

    TaskDispatcher dispatcher(sourceArgs.size(), arguments);
    const auto params = dispatcher.parseInputArgs();
    EXPECT_EQ(params.getInterpolationMethod(), InterpolationMethod::Trilinear);
}

struct IncorrectDimensionsTest : public InputArgsParserTest, public ::testing::WithParamInterface<std::tuple<std::vector<std::string>, int, int, bool>> {};

TEST_P(IncorrectDimensionsTest, incorrectDimensions) {
    const auto&[additionalArgs, expectedWidth, expectedHeight, shouldFail] = GetParam();
    std::vector<std::string> arguments{"program", "-i", "abc.png", "-l", "test.cube"};
    arguments.insert(arguments.end(), additionalArgs.begin(), additionalArgs.end());
    auto rawArgumentsArray = initialize2DArray(arguments);
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
    const auto arguments = initialize2DArray(sourceArgs);

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
            std::vector<std::string>{"-t"},
            std::vector<std::string>{"-l", "-t", "-i"},
            std::vector<std::string>{"-l", "abcd.cube", "-i", "abcd.png", "--abc"},
            std::vector<std::string>{"-l", "abcd.cube", "-i", "abcd.png", "--strength"},
            std::vector<std::string>{"-l", "abcd.cube", "-i", "abcd.png", "--strength="},
            std::vector<std::string>{"-l", "abcd.cube", "-i", "abcd.png", "--strength", "-t"}
        )
);
