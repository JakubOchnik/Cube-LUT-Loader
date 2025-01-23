#pragma once
#include <Eigen/Dense>
#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/CPU/LUT3DPipelineCPU.hpp>
#include <ImageProcessing/CPU/WorkerData.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

class NearestValImplCPU : public LUT3DPipelineCPU {
public:
	using LUT3DPipelineCPU::LUT3DPipelineCPU;

private:
	void calculatePixel(const int x, const int y, const Table3D& lut, const WorkerData& data) override;
};
