#pragma once
#include <FileIO/FileIO.hpp>
#include <opencv2/core.hpp>

class ImageProcessExecutor
{
protected:
    cv::Mat3b newImg;
    FileIO& fileInterface;

public:
    ImageProcessExecutor(FileIO &fileIfc);
    ImageProcessExecutor() = delete;
    virtual ~ImageProcessExecutor() = default;

    virtual cv::Mat process(float intensity, InterpolationMethod method) = 0;
    cv::Mat execute(float intensity, cv::Size dstImageSize, InterpolationMethod method);
    void resizeImage(cv::Mat3b& inputImg, cv::Size size, cv::Mat1b& inputAlpha, int interpolationMode = 2);
};
