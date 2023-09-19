#pragma once
#include <DataLoader/DataLoader.hpp>
#include <opencv2/core.hpp>

class ImageProcessor
{
protected:
    cv::Mat_<cv::Vec3b> newImg;
    const DataLoader &loader;

public:
    ImageProcessor(const DataLoader &ld);
    ImageProcessor() = delete;

    virtual cv::Mat process() = 0;
    virtual void save() const;
    virtual void execute();
};
