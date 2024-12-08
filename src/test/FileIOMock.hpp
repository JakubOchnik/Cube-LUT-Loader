#pragma once
#include <gmock/gmock.h>
#include <FileIO/FileIO.hpp>
#include <CubeLUTMock.hpp>
#include <system_error>

class FileIOMock : public FileIO {
public:
    using FileIO::FileIO;
    using FileIO::loadLut;
    using FileIO::loadImg;
public:
    MOCK_METHOD(cv::Mat, readImage, (const std::string&), (const, override));
    MOCK_METHOD(bool, writeImage, (const std::string&, cv::Mat), (const, override));
    MOCK_METHOD(bool, fileExists, (std::error_code& ec), (const, override));

    void setCube(std::unique_ptr<BasicCubeLUTMock>&& newCube) {
        cube = std::move(newCube);
    }

    cv::Mat1b getAlphaChannel() {
        return alphaChannel;
    }
};
