#pragma once
#include <ImageProcessing/GPU/LUT3DPipelineGPU.hpp>

class TrilinearImplGPU : public LUT3DPipelineGPU {
public:
    using LUT3DPipelineGPU::LUT3DPipelineGPU;

private:
    void runKernel(dim3 threads, dim3 blocks, unsigned char *image, const char channels, float *LUT, const int LUTsize,
			 const float opacity, const std::tuple<int, int> &imgSize) override;
};