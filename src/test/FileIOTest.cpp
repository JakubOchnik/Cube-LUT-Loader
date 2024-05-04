#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <FileIOMock.hpp>
#include <CubeLUTMock.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace ::testing;

class FileIOTest : public ::testing::Test {
protected:
    unsigned int defaultWidth = 2;
    unsigned int defaultHeight = 2;
    cv::Mat testMat = cv::Mat(defaultWidth, defaultHeight, CV_8UC3);
};

TEST_F(FileIOTest, testLoadImage) {
    FileIOMock loader(InputParams{});

    EXPECT_CALL(loader, readImage).WillOnce(Return(testMat));
    EXPECT_TRUE(loader.loadImg());
}

TEST_F(FileIOTest, testLoadImageIncorrectPath) {
    FileIOMock loader(InputParams{});

    EXPECT_CALL(loader, readImage).WillOnce(Return(cv::Mat()));
    EXPECT_FALSE(loader.loadImg());
}

struct ImageResizeTest : public ::testing::WithParamInterface<std::pair<int, int>>, public FileIOTest {};

TEST_P(ImageResizeTest, clippingTest) {
    const auto [width, height] =  GetParam();
    InputParams params;
    unsigned int expectedWidth = defaultWidth;
    if (width) {
        params.setOutputImageWidth(width);
        expectedWidth = width;
    }
    unsigned int expectedHeight = defaultHeight;
    if (height) {
        params.setOutputImageHeight(height);
        expectedHeight = height;
    }
    FileIOMock loader(params);

    EXPECT_CALL(loader, readImage).WillOnce(Return(testMat));
    EXPECT_CALL(loader, resizeImage(_, _, expectedWidth, expectedHeight, cv::InterpolationFlags::INTER_CUBIC));
    EXPECT_TRUE(loader.loadImg());
}

INSTANTIATE_TEST_SUITE_P(
        FileIOTest,
        ImageResizeTest,
        ::testing::Values(
            std::make_pair(10,0),
            std::make_pair(0,10),
            std::make_pair(10,10)
        )
);

TEST_F(FileIOTest, testLoadLutIncorrectPath) {
    FileIOMock loader(InputParams{});
    BasicCubeLUTMock* lutMock = new BasicCubeLUTMock;
    EXPECT_CALL(*lutMock, loadCubeFile).Times(0);
    loader.setCube(lutMock);
    EXPECT_FALSE(loader.loadLut());
}

TEST_F(FileIOTest, testLoadLut) {
    constexpr auto lutPath = "resources/alog.cube";
    InputParams params;
    params.setInputLutPath(lutPath);
    FileIOMock loader(params);
    BasicCubeLUTMock* lutMock = new BasicCubeLUTMock;
    EXPECT_CALL(*lutMock, loadCubeFile).Times(1);
    loader.setCube(lutMock);
    EXPECT_TRUE(loader.loadLut());
}

TEST_F(FileIOTest, testLoadLutFailed) {
    constexpr auto lutPath = "resources/alog.cube";
    InputParams params;
    params.setInputLutPath(lutPath);
    FileIOMock loader(params);
    BasicCubeLUTMock* lutMock = new BasicCubeLUTMock;
    EXPECT_CALL(*lutMock, loadCubeFile).WillOnce(Throw(std::runtime_error("TEST")));
    loader.setCube(lutMock);
    EXPECT_FALSE(loader.loadLut());
}
