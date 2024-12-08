#include <gtest/gtest.h>
#include <FileIOMock.hpp>
#include <CPUProcessorMock.hpp>

using namespace ::testing;

class ImageProcessorTest : public ::testing::Test {
protected:    
    const int DEFAULT_WIDTH = 2;
    const int DEFAULT_HEIGHT = 2;
    cv::Mat3b mockImage{cv::Size(DEFAULT_WIDTH, DEFAULT_HEIGHT), CV_8UC3};
};

struct ImageResizeTest : public ::testing::WithParamInterface<std::pair<int, int>>, public ImageProcessorTest {};

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
        ImageProcessorTest,
        ImageResizeTest,
        ::testing::Values(
            std::make_pair(0,0),
            std::make_pair(10,0),
            std::make_pair(0,10),
            std::make_pair(10,10)
        )
);
