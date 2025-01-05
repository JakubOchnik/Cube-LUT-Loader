#pragma once
#include <FileIO/CubeLUT.hpp>
#include <opencv2/core.hpp>
#include <variant>

class LUTProcessor {
public:
	virtual cv::Mat execute(cv::Mat img, const float opacity, const uint threadPool) = 0;

	virtual ~LUTProcessor() = default;
};
