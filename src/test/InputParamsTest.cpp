#include <gtest/gtest.h>
#include <TaskDispatcher/InputParams.h>
#include <boost/program_options.hpp>

using namespace ::testing;

class InputParamsTest : public ::testing::Test {};

void testDefaultValues(const InputParams& params) {
    EXPECT_EQ(params.getEffectStrength(), 1u);
    EXPECT_EQ(params.getThreads(), 1u);
    EXPECT_EQ(params.getProcessingMode(), ProcessingMode::CPU);
    EXPECT_EQ(params.getShowHelp(), false);
    EXPECT_EQ(params.getOutputImageWidth(), 0);
    EXPECT_EQ(params.getOutputImageHeight(), 0);
}

TEST_F(InputParamsTest, testDefaultValues)
{
    InputParams params;
    testDefaultValues(params);
}

TEST_F(InputParamsTest, doNothingWhenNoValue)
{
    boost::program_options::variables_map vm;
    InputParams params(std::move(vm));
    testDefaultValues(params);
}

TEST_F(InputParamsTest, testParsingValues)
{
    boost::program_options::variables_map vm;
    vm.emplace("gpu", boost::program_options::variable_value(true, false));
    vm.emplace("help", boost::program_options::variable_value(true, false));
    vm.emplace("input", boost::program_options::variable_value(std::string{"test_input"}, false));
    vm.emplace("output", boost::program_options::variable_value(std::string{"test_output"}, false));
    vm.emplace("lut", boost::program_options::variable_value(std::string{"test_lut"}, false));
    vm.emplace("strength", boost::program_options::variable_value(.5f, false));
    vm.emplace("threads", boost::program_options::variable_value(16u, false));
    vm.emplace("nearest_value", boost::program_options::variable_value(true, false));
    vm.emplace("width", boost::program_options::variable_value(1337, false));
    vm.emplace("height", boost::program_options::variable_value(420, false));

    InputParams params(std::move(vm));
    EXPECT_EQ(params.getProcessingMode(), ProcessingMode::GPU);
    EXPECT_EQ(params.getShowHelp(), true);
    EXPECT_EQ(params.getInputImgPath(), "test_input");
    EXPECT_EQ(params.getOutputImgPath(), "test_output");
    EXPECT_EQ(params.getInputLutPath(), "test_lut");
    EXPECT_EQ(params.getEffectStrength(), .5f);
    EXPECT_EQ(params.getThreads(), 16u);
    EXPECT_EQ(params.getInterpolationMethod(), InterpolationMethod::NearestValue);
    EXPECT_EQ(params.getOutputImageWidth(), 1337);
    EXPECT_EQ(params.getOutputImageHeight(), 420);
}

TEST_F(InputParamsTest, testAmbiguousInterpolationInput)
{
    boost::program_options::variables_map vm;
    vm.emplace("nearest_value", boost::program_options::variable_value(true, false));
    vm.emplace("trilinear", boost::program_options::variable_value(true, false));

    InputParams params(std::move(vm));
    EXPECT_EQ(params.getInterpolationMethod(), InterpolationMethod::Trilinear);
}

TEST_F(InputParamsTest, testIncorrectImageOutputSize)
{
    boost::program_options::variables_map vm;
    vm.emplace("width", boost::program_options::variable_value(-10, false));
    EXPECT_THROW(InputParams(std::move(vm)), std::runtime_error);

    vm.clear();
    vm.emplace("height", boost::program_options::variable_value(0, false));
    EXPECT_THROW(InputParams(std::move(vm)), std::runtime_error);
}
