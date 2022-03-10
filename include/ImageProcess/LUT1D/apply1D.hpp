#pragma once
#include <opencv2/opencv.hpp>
#include <Loader/CubeLUT.hpp>

cv::Mat_<cv::Vec3b> applyBasic1D(const cv::Mat& img, const CubeLUT& lut, float opacity);

double getAvgVal(const CubeLUT& lut, unsigned int nValues, unsigned char value, unsigned char channel);

unsigned char getColor(double value);