#pragma once
#include <Eigen/Core>
#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/CPU/LUT3DPipelineCPU.hpp>
#include <ImageProcessing/CPU/WorkerData.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

class TetrahedralImplCPU : public LUT3DPipelineCPU {
public:
    explicit TetrahedralImplCPU(Table3D* lut);

private:
	void calculatePixel(const int x, const int y, const Table3D& lut, const WorkerData& data) override;

    std::vector<Eigen::Matrix<float, 4, 8>> matrices;
    float lutScale;
};
