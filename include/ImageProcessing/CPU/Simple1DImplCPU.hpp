#pragma once
#include <Eigen/Dense>
#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/CPU/WorkerData.hpp>
#include <ImageProcessing/LUTProcessor.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

class Simple1DImplCPU : public LUTProcessor {
public:
	Simple1DImplCPU(Table1D* lut);
	virtual cv::Mat execute(cv::Mat img, const float opacity, const uint threadPool) override;

private:
	Table1D* lut1d;

	void calculatePixel(const int x, const int y, const Table1D& lut, const WorkerData& data);
	void calculateArea(const int x, const Table1D& lut, const WorkerData& data, const int segWidth);
};
