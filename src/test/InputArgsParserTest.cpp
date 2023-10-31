#include <gtest/gtest.h>
#include <TaskDispatcher/TaskDispatcher.hpp>

using namespace ::testing;

class InputArgsParserTest : public ::testing::Test {};

char** initialize2DArray(const std::vector<std::string>& args) {
    char** arr = new char*[args.size() + 1];
    for (int i{}; i < args.size(); ++i) {
        const auto& argument = args[i];
        arr[i] = new char[argument.size()];
        std::strcpy(arr[i], argument.c_str());
    }
    arr[args.size()] = nullptr;
    return arr;
}

void free2Darray(char** arr, unsigned int size) {
    for (int i{}; i < size + 1; ++i) {
        delete arr[i];
    }
    delete arr;
}

TEST_F(InputArgsParserTest, testHelp)
{
    const std::vector<std::string> sourceArgs{"program", "-h"};
    const auto arguments = initialize2DArray(sourceArgs);
    const auto argCount = sourceArgs.size();

    TaskDispatcher dispatcher(argCount, arguments);
    std::string helpText;
    const auto params = dispatcher.parseInputArgs(helpText);
    EXPECT_EQ(params.getShowHelp(), true);
    EXPECT_GT(helpText.size(), 0);

    free2Darray(arguments, argCount);
}

TEST_F(InputArgsParserTest, emptyArgs)
{
    const std::vector<std::string> sourceArgs{"program"};
    const auto arguments = initialize2DArray(sourceArgs);
    const auto argCount = sourceArgs.size();

    TaskDispatcher dispatcher(argCount, arguments);
    std::string helpText;
    EXPECT_THROW(dispatcher.parseInputArgs(helpText), boost::program_options::error);

    free2Darray(arguments, argCount);
}

TEST_F(InputArgsParserTest, multipleInterpolationMethods)
{
    const std::vector<std::string> sourceArgs{"program", "-i", "abc.png", "-l", "test.cube", "-t", "-n"};
    const auto arguments = initialize2DArray(sourceArgs);
    const auto argCount = sourceArgs.size();

    TaskDispatcher dispatcher(argCount, arguments);
    std::string helpText;
    const auto params = dispatcher.parseInputArgs(helpText);
    EXPECT_EQ(params.getInterpolationMethod(), InterpolationMethod::Trilinear);

    free2Darray(arguments, argCount);
}

struct IncorrectInputTest : public ::testing::TestWithParam<std::vector<std::string>> {};

TEST_P(IncorrectInputTest, incorrectParam) {
    std::vector<std::string> sourceArgs{"program"};
    const auto& testArgs = GetParam();
    sourceArgs.insert(sourceArgs.end(), testArgs.begin(), testArgs.end());
    const auto arguments = initialize2DArray(sourceArgs);
    const auto argCount = sourceArgs.size();

    TaskDispatcher dispatcher(argCount, arguments);
    std::string helpText;
    EXPECT_THROW(dispatcher.parseInputArgs(helpText), boost::program_options::error);

    free2Darray(arguments, argCount);
}

INSTANTIATE_TEST_SUITE_P(
        InputArgsParserTest,
        IncorrectInputTest,
        ::testing::Values(
            std::vector<std::string>{},
            std::vector<std::string>{"-l", "abcd.cube"},
            std::vector<std::string>{"-i", "abcd.png"},
            std::vector<std::string>{"-t"}
        )
);
