#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <FileIOMock.hpp>
#include <CubeLUTMock.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace ::testing;

class FileIOTest : public ::testing::Test {
protected:
    int defaultWidth = 2;
    int defaultHeight = 2;
    cv::Mat testMat = cv::Mat(defaultWidth, defaultHeight, CV_8UC3);

    cv::Mat getInitializedMatWithAlpha() {
        std::vector<cv::Mat1b> channels;
        channels.reserve(4);
        channels.emplace_back(defaultWidth, defaultHeight, 0); // b
        channels.emplace_back(defaultWidth, defaultHeight, 1); // g
        channels.emplace_back(defaultWidth, defaultHeight, 2); // r
        channels.emplace_back(defaultWidth, defaultHeight, 3); // a
        cv::Mat dstMat;
        cv::merge(channels, dstMat);
        return dstMat;
    }

    InputParams params;
};

TEST_F(FileIOTest, testLoadImage) {
    FileIOMock loader(InputParams{});

    EXPECT_CALL(loader, readImage).WillOnce(Return(testMat));
    EXPECT_TRUE(loader.loadImg());
    EXPECT_TRUE(loader.getAlpha().empty());
}

namespace {
    void compareChannel(cv::Mat mat, int channelNumber, const std::vector<uchar>& values) {
        cv::Mat1b channel;
        cv::extractChannel(mat, channel, channelNumber);
        ASSERT_EQ(channel.rows * channel.cols, values.size());
        const auto* matData = channel.data;
        for (int i{}; i < values.size(); ++i) {
            EXPECT_EQ(values[i], matData[i]);
        }
    }
}

TEST_F(FileIOTest, loadImageWithAlpha) {
    FileIOMock loader(InputParams{});

    const auto exampleAlphaMat = getInitializedMatWithAlpha();
    EXPECT_CALL(loader, readImage).WillOnce(Return(exampleAlphaMat));
    ASSERT_TRUE(loader.loadImg());

    const auto& alphaChannel = loader.getAlpha();
    EXPECT_FALSE(alphaChannel.empty());
    compareChannel(alphaChannel, 0, { 3, 3, 3, 3 });

    const auto& img = loader.getImg();
    EXPECT_EQ(img.channels(), 3);
    compareChannel(img, 0, { 0, 0, 0, 0 });
    compareChannel(img, 1, { 1, 1, 1, 1 });
    compareChannel(img, 2, { 2, 2, 2, 2 });
}

TEST_F(FileIOTest, writeImageWithAlpha) {
    FileIOMock loader(InputParams{});

    const auto exampleAlphaMat = getInitializedMatWithAlpha();
    EXPECT_CALL(loader, readImage).WillOnce(Return(exampleAlphaMat));
    ASSERT_TRUE(loader.loadImg());

    const auto& alphaChannel = loader.getAlpha();
    EXPECT_FALSE(alphaChannel.empty());
    compareChannel(alphaChannel, 0, { 3, 3, 3, 3 });

    const auto& img = loader.getImg();
    EXPECT_EQ(img.channels(), 3);
    compareChannel(img, 0, { 0, 0, 0, 0 });
    compareChannel(img, 1, { 1, 1, 1, 1 });
    compareChannel(img, 2, { 2, 2, 2, 2 });
}

TEST_F(FileIOTest, testLoadImageIncorrectPath) {
    FileIOMock loader(InputParams{});

    EXPECT_CALL(loader, readImage).WillOnce(Return(cv::Mat()));
    EXPECT_FALSE(loader.loadImg());
}

TEST_F(FileIOTest, testLoadLutIncorrectPath) {
    FileIOMock loader(params);
    std::unique_ptr<BasicCubeLUTMock> lutMock = std::make_unique<BasicCubeLUTMock>();
    EXPECT_CALL(*lutMock, loadCubeFile).Times(0);
    loader.setCube(std::move(lutMock));
    EXPECT_FALSE(loader.loadLut());
}

TEST_F(FileIOTest, testLoadLut) {
    constexpr auto lutPath = "resources/alog.cube";
    params.setInputLutPath(lutPath);
    FileIOMock loader(params);
    std::unique_ptr<BasicCubeLUTMock> lutMock = std::make_unique<BasicCubeLUTMock>();
    EXPECT_CALL(*lutMock, loadCubeFile).Times(1);
    loader.setCube(std::move(lutMock));
    EXPECT_TRUE(loader.loadLut());
}

TEST_F(FileIOTest, testLoadLutFailed) {
    constexpr auto lutPath = "resources/alog.cube";
    params.setInputLutPath(lutPath);
    FileIOMock loader(params);
    std::unique_ptr<BasicCubeLUTMock> lutMock = std::make_unique<BasicCubeLUTMock>();
    EXPECT_CALL(*lutMock, loadCubeFile).WillOnce(Throw(std::runtime_error("TEST")));
    loader.setCube(std::move(lutMock));
    EXPECT_FALSE(loader.loadLut());
}


TEST_F(FileIOTest, saveWithForceOverwrite) {
    params.setForceOverwrite(true);
    FileIOMock loader(params);
    EXPECT_CALL(loader, fileExists).Times(0);
    EXPECT_CALL(loader, writeImage).WillOnce(Return(true));
    EXPECT_TRUE(loader.saveImg({}));

    EXPECT_CALL(loader, fileExists).Times(0);
    EXPECT_CALL(loader, writeImage).WillOnce(Return(false));
    EXPECT_FALSE(loader.saveImg({}));
}

TEST_F(FileIOTest, saveWithoutForceOverwrite) {
    params.setForceOverwrite(false);
    FileIOMock loader(params);
    EXPECT_CALL(loader, fileExists).WillOnce(Invoke([](std::error_code& ec){
        return true;
    }));
    EXPECT_CALL(loader, writeImage).Times(0);
    EXPECT_FALSE(loader.saveImg({}));

    EXPECT_CALL(loader, fileExists).WillOnce(Invoke([](std::error_code& ec){
        ec = std::make_error_code(std::errc::read_only_file_system);
        return false;
    }));
    EXPECT_CALL(loader, writeImage).Times(0);
    EXPECT_FALSE(loader.saveImg({}));

    EXPECT_CALL(loader, fileExists).WillOnce(Return(false));
    EXPECT_CALL(loader, writeImage).WillOnce(Return(true));
    EXPECT_TRUE(loader.saveImg({}));
}

TEST_F(FileIOTest, readHighBitDepth) {
    FileIOMock loader(params);
    cv::Mat mat16(cv::Size(10, 10), CV_16UC3);
    EXPECT_CALL(loader, readImage).WillOnce(Return(mat16));
    EXPECT_TRUE(loader.loadImg());
    const auto parsedMat16 = loader.getImg();
    EXPECT_EQ(parsedMat16.depth(), CV_8U);
    EXPECT_EQ(parsedMat16.channels(), 3);
    EXPECT_EQ(parsedMat16.type(), CV_8UC3);

    cv::Mat mat16alpha(cv::Size(10, 10), CV_16UC4);
    EXPECT_CALL(loader, readImage).WillOnce(Return(mat16alpha));
    EXPECT_TRUE(loader.loadImg());
    const auto parsedMat16alpha = loader.getImg();
    EXPECT_EQ(parsedMat16alpha.depth(), CV_8U);
    EXPECT_EQ(parsedMat16alpha.channels(), 3);
    EXPECT_EQ(parsedMat16alpha.type(), CV_8UC3);
    EXPECT_FALSE(loader.getAlpha().empty());
}
