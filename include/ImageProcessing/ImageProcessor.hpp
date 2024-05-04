#pragma once
#include <FileIO/FileIO.hpp>
#include <opencv2/core.hpp>

class ImageProcessor
{
protected:
    cv::Mat_<cv::Vec3b> newImg;
    const FileIO &loader;

public:
    ImageProcessor(const FileIO &ld);
    ImageProcessor() = delete;

    virtual cv::Mat process() = 0;
    virtual void save() const;
    virtual void execute();
};
