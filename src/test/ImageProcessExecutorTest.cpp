#include <gtest/gtest.h>
#include <FileIOMock.hpp>
#include <CPUProcessorMock.hpp>

using namespace ::testing;

class ImageProcessExecutorTest : public ::testing::Test {
protected:
    const int DEFAULT_WIDTH = 2;
    const int DEFAULT_HEIGHT = 2;
    cv::Mat3b mockImage{cv::Size(DEFAULT_WIDTH, DEFAULT_HEIGHT), CV_8UC3};
    cv::Mat1b mockAlpha{cv::Size(DEFAULT_WIDTH, DEFAULT_HEIGHT)};
};

TEST_F(ImageProcessExecutorTest, scaleAlpha) {
    constexpr const int dstWidth = 42;
    constexpr const int dstHeight = 10;
    FileIOMock fileIO(InputParams{});
    fileIO.setImg(mockImage);
    fileIO.setAlpha(mockAlpha);
    CPUProcessorMock imProc(fileIO, 1);
    EXPECT_CALL(imProc, process).Times(1);
    imProc.execute(1.0f, {dstWidth, dstHeight}, InterpolationMethod::NearestValue);
    EXPECT_EQ(fileIO.getImg().cols, dstWidth);
    EXPECT_EQ(fileIO.getImg().rows, dstHeight);
    EXPECT_EQ(fileIO.getAlpha().cols, dstWidth);
    EXPECT_EQ(fileIO.getAlpha().rows, dstHeight);
}

struct ImageResizeTest : public ::testing::WithParamInterface<std::pair<int, int>>, public ImageProcessExecutorTest {};

TEST_P(ImageResizeTest, clippingTest) {
    const auto [width, height] =  GetParam();
    InputParams params;
    int expectedWidth = DEFAULT_WIDTH;
    if (width) {
        params.setOutputImageWidth(width);
        expectedWidth = width;
    }
    int expectedHeight = DEFAULT_HEIGHT;
    if (height) {
        params.setOutputImageHeight(height);
        expectedHeight = height;
    }
    FileIOMock fileIO(params);
    fileIO.setImg(mockImage);
    CPUProcessorMock imProc(fileIO, 1);
    EXPECT_CALL(imProc, process).Times(1);
    imProc.execute(1.0f, {width, height}, InterpolationMethod::NearestValue);
    EXPECT_EQ(fileIO.getImg().cols, expectedWidth);
    EXPECT_EQ(fileIO.getImg().rows, expectedHeight);
}

INSTANTIATE_TEST_SUITE_P(
        ImageProcessExecutorTest,
        ImageResizeTest,
        ::testing::Values(
            std::make_pair(0,0),
            std::make_pair(10,0),
            std::make_pair(0,10),
            std::make_pair(10,10)
        )
);
