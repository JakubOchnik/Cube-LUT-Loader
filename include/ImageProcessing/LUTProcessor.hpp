#pragma once
#include <FileIO/CubeLUT.hpp>
#include <opencv2/core.hpp>
#include <variant>

class LUTProcessor {
public:
	explicit LUTProcessor(Table1D* lut) : lut1d(lut) {}
	explicit LUTProcessor(Table3D* lut) : lut3d(lut) {}

	virtual cv::Mat execute(cv::Mat img, const float opacity, const uint threadPool) = 0;

	virtual ~LUTProcessor() = default;

protected:
	Table1D* lut1d{nullptr};
	Table3D* lut3d{nullptr};
};
