#pragma once
#include <opencv2/opencv.hpp>
#include "../Loader/CubeLUT.hpp"

cv::Mat_<cv::Vec3b> applyNearest(cv::Mat img, CubeLUT lut, float opacity);