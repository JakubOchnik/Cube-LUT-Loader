#pragma once
#include <FileIO/FileIO.hpp>
#include <opencv2/core.hpp>

class ImageProcessExecutor
{
protected:
    cv::Mat_<cv::Vec3b> newImg;
    FileIO& fileInterface;

public:
    ImageProcessExecutor(FileIO &fileIfc);
    ImageProcessExecutor() = delete;
    virtual ~ImageProcessExecutor() = default;

    virtual cv::Mat process(float intensity, InterpolationMethod method) = 0;
    virtual cv::Mat execute(float intensity, cv::Size dstImageSize, InterpolationMethod method);
    virtual cv::Mat resizeImage(cv::Mat inputImg, cv::Size size, int interpolationMode = 2);
};
