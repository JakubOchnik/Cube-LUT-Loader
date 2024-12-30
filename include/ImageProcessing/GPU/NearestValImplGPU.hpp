#pragma once
#include <ImageProcessing/GPU/LUT3DPipelineGPU.hpp>

class NearestValImplGPU : public LUT3DPipelineGPU {
public:
    using LUT3DPipelineGPU::LUT3DPipelineGPU;

private:
    virtual void runKernel(dim3 threads, dim3 blocks, unsigned char *image, const char channels, float *LUT, const int LUTsize,
			 const float opacity, const std::tuple<int, int> &imgSize) override;
};