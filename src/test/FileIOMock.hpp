#pragma once
#include <gmock/gmock.h>
#include <FileIO/FileIO.hpp>
#include <CubeLUTMock.hpp>

class FileIOMock : public FileIO {
public:
    using FileIO::FileIO;
public:
    MOCK_METHOD(cv::Mat, readImage, (const std::string&), (override));
    MOCK_METHOD(void, resizeImage, (cv::Mat, cv::Mat, unsigned int, unsigned ing, int), (override));

    void setCube(BasicCubeLUTMock* newCube) {
        cube.reset(newCube);
    }
};
