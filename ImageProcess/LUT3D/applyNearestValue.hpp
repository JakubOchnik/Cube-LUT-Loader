#pragma once
#include <opencv2/opencv.hpp>
#include "../../Loader/CubeLUT.hpp"

cv::Mat applyNearest(cv::Mat img, CubeLUT lut, float opacity);