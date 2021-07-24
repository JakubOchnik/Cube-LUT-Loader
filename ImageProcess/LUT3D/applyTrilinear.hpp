#pragma once
#include <opencv2/opencv.hpp>
#include "../../Loader/CubeLUT.hpp"

cv::Mat_<cv::Vec3b> applyTrilinear(cv::Mat img, CubeLUT lut, float opacity);

vector<float> mul(const vector<float>& vec, const float val);

vector<float> sum(const vector<float>& a, const vector<float>& b);