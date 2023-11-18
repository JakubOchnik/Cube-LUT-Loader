#pragma once
#include <gmock/gmock.h>
#include <DataLoader/DataLoader.hpp>
#include <CubeLUTMock.hpp>

class DataLoaderMock : public DataLoader {
public:
    using DataLoader::DataLoader;
public:
    MOCK_METHOD(cv::Mat, readImage, (const std::string&), (override));
    MOCK_METHOD(void, resizeImage, (cv::Mat, cv::Mat, unsigned int, unsigned ing, int), (override));

    void setCube(BasicCubeLUTMock* newCube) {
        cube.reset(newCube);
    }
};
