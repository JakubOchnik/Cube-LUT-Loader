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
    InputParams params;
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
