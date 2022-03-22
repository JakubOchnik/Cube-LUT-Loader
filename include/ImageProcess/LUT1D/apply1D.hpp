#pragma once
#include <opencv2/opencv.hpp>
#include <Loader/CubeLUT.hpp>


namespace Basic1D
{
    struct WorkerData
	{
		unsigned char* image;
		unsigned char* new_image;
		const int width;
		const int height;
		const int channels;
		const int lutSize;
        const int nValues;
	};

    cv::Mat_<cv::Vec3b> applyBasic1D(const cv::Mat& img, const CubeLUT& lut, float opacity, const uint threadPool);
    void calculateArea(const int x, const CubeLUT& lut, const float opacity, WorkerData& data, const int segWidth);
    void calculatePixel(const int x, const int y, const CubeLUT& lut, const float opacity, WorkerData& data);

    double getAvgVal(const CubeLUT& lut, const uint nValues, const uchar value, const uchar channel);
    uchar getColor(double value);

}
