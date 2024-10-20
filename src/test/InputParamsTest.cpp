#include <gtest/gtest.h>
#include <TaskDispatcher/InputParams.h>
#include <args.hxx>

using namespace ::testing;

class InputParamsTest : public ::testing::Test {};

void testDefaultValues(const InputParams& params) {
    EXPECT_EQ(params.getEffectStrength(), 1u);
    EXPECT_EQ(params.getThreads(), 1u);
    EXPECT_EQ(params.getProcessingMode(), ProcessingMode::CPU);
    EXPECT_EQ(params.getForceOverwrite(), false);
    EXPECT_EQ(params.getOutputImageWidth(), 0);
    EXPECT_EQ(params.getOutputImageHeight(), 0);
}

TEST_F(InputParamsTest, testDefaultValues)
{
    InputParams params;
    testDefaultValues(params);
}

TEST_F(InputParamsTest, mapFlagsToProcessingmode) {
    EXPECT_EQ(flagsToProcessingMode(true), ProcessingMode::GPU);
    EXPECT_EQ(flagsToProcessingMode(false), ProcessingMode::CPU);
}

TEST_F(InputParamsTest, mapFlagsToInterpolationMethod) {
    EXPECT_EQ(flagsToInterpolationMethod(true, true), InterpolationMethod::Trilinear);
    EXPECT_EQ(flagsToInterpolationMethod(true, false), InterpolationMethod::Trilinear);
    EXPECT_EQ(flagsToInterpolationMethod(false, false), InterpolationMethod::Trilinear);
    EXPECT_EQ(flagsToInterpolationMethod(false, true), InterpolationMethod::NearestValue);
}

TEST_F(InputParamsTest, testParsingValues)
{
    InputParams params {
        ProcessingMode::GPU,
        16u,
        InterpolationMethod::NearestValue,
        "test_input",
        "test_output",
        true,
        "test_lut",
        50.0f,
        1337,
        420
    };

    EXPECT_EQ(params.getProcessingMode(), ProcessingMode::GPU);
    EXPECT_EQ(params.getForceOverwrite(), true);
    EXPECT_EQ(params.getInputImgPath(), "test_input");
    EXPECT_EQ(params.getOutputImgPath(), "test_output");
    EXPECT_EQ(params.getInputLutPath(), "test_lut");
    EXPECT_EQ(params.getEffectStrength(), .5f);
    EXPECT_EQ(params.getThreads(), 16u);
    EXPECT_EQ(params.getInterpolationMethod(), InterpolationMethod::NearestValue);
    EXPECT_EQ(params.getOutputImageWidth(), 1337);
    EXPECT_EQ(params.getOutputImageHeight(), 420);
}
