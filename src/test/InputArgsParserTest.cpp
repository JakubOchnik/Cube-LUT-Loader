#include <gtest/gtest.h>
#include <TaskDispatcher/TaskDispatcher.hpp>
#include <string.h>

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
        if (argCount <= 0) {
            return;
        }

        for (int i{}; i < argCount - 1; ++i) {
            delete arguments[i];
        }
        delete arguments;
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
    std::string helpText;
    const auto params = dispatcher.parseInputArgs(helpText);
    EXPECT_EQ(params.getShowHelp(), true);
    EXPECT_GT(helpText.size(), 0);
}

TEST_F(InputArgsParserTest, emptyArgs)
{
    const std::vector<std::string> sourceArgs{"program"};
    const auto arguments = initialize2DArray(sourceArgs);

    TaskDispatcher dispatcher(sourceArgs.size(), arguments);
    std::string helpText;
    EXPECT_THROW(dispatcher.parseInputArgs(helpText), boost::program_options::error);
}

TEST_F(InputArgsParserTest, multipleInterpolationMethods)
{
    const std::vector<std::string> sourceArgs{"program", "-i", "abc.png", "-l", "test.cube", "-t", "-n"};
    const auto arguments = initialize2DArray(sourceArgs);

    TaskDispatcher dispatcher(sourceArgs.size(), arguments);
    std::string helpText;
    const auto params = dispatcher.parseInputArgs(helpText);
    EXPECT_EQ(params.getInterpolationMethod(), InterpolationMethod::Trilinear);
}

struct IncorrectInputTest : public InputArgsParserTest, public ::testing::WithParamInterface<std::vector<std::string>> {};

TEST_P(IncorrectInputTest, incorrectParam) {
    std::vector<std::string> sourceArgs{"program"};
    const auto& testArgs = GetParam();
    sourceArgs.insert(sourceArgs.end(), testArgs.begin(), testArgs.end());
    const auto arguments = initialize2DArray(sourceArgs);

    TaskDispatcher dispatcher(sourceArgs.size(), arguments);
    std::string helpText;
    EXPECT_THROW(dispatcher.parseInputArgs(helpText), boost::program_options::error);
}

INSTANTIATE_TEST_SUITE_P(
        InputArgsParserTest,
        IncorrectInputTest,
        ::testing::Values(
            std::vector<std::string>{},
            std::vector<std::string>{"-l", "abcd.cube"},
            std::vector<std::string>{"-i", "abcd.png"},
            std::vector<std::string>{"-t"},
            std::vector<std::string>{"-l", "-t", "-i"}
        )
);
