#pragma once
#include <ImageProcessing/CPU/WorkerData.hpp>
#include <ImageProcessing/LUTProcessor.hpp>

class LUT3DPipelineCPU : public LUTProcessor {
public:
	LUT3DPipelineCPU(Table3D* lut);
	virtual cv::Mat execute(cv::Mat img, const float opacity, const uint threadPool) override;

protected:
	Table3D* lut3d{nullptr};

	virtual void calculatePixel(const int x, const int y, const Table3D& lut, const WorkerData& data) {}
	virtual void calculateArea(const int x, const Table3D& lut, const WorkerData& data, const int segWidth) {}
};
