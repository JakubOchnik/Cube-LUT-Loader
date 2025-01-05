#include <ImageProcessing/CPU/LUT3DPipelineCPU.hpp>
#include <ImageProcessing/CPU/Simple1DImplCPU.hpp>
#include <gtest/gtest.h>

#if BUILD_CUDA
#include <CudaCallsMock.hpp>
#include <ImageProcessing/GPU/LUT3DPipelineGPU.hpp>
#endif

#include <FileIO/CubeLUT.hpp>

using namespace ::testing;

class LUTProcessorTest : public ::testing::Test {
#if BUILD_CUDA
protected:
	CudaCallsMock calls;
#endif
};

TEST_F(LUTProcessorTest, incorrectLutPassedCPU) {
	cv::Mat img;
	Table1D* lut1d = nullptr;
	Simple1DImplCPU processor1d(lut1d);
	EXPECT_THROW(processor1d.execute(img, 1.0f, 1u), std::runtime_error);

	Table3D* lut3d = nullptr;
	LUT3DPipelineCPU processor(lut3d);
	EXPECT_THROW(processor.execute(img, 1.0f, 1u), std::runtime_error);

	Table3D sample3dLut;
	lut3d = &sample3dLut;
	processor = LUT3DPipelineCPU(lut3d);
	EXPECT_NO_THROW(processor.execute(img, 1.0f, 1u));

	Table1D sample1dLut;
	lut1d = &sample1dLut;
	processor1d = Simple1DImplCPU(lut1d);
	EXPECT_NO_THROW(processor1d.execute(img, 1.0f, 1u));
}

#if BUILD_CUDA
TEST_F(LUTProcessorTest, incorrectLutPassedGPU) {
	cv::Mat img;
	Table3D* lut3d = nullptr;
	LUT3DPipelineGPU processor = LUT3DPipelineGPU(lut3d, &calls);
	EXPECT_THROW(processor.execute(img, 1.0f, 1u), std::runtime_error);

	Table3D sampleLut;
	lut3d = &sampleLut;
	processor = LUT3DPipelineGPU(lut3d, &calls);
	EXPECT_NO_THROW(processor.execute(img, 1.0f, 1u));
}

TEST_F(LUTProcessorTest, nullCudaCalls) {
	cv::Mat img;
	Table3D lut3d;
	LUT3DPipelineGPU processor = LUT3DPipelineGPU(&lut3d, nullptr);
	EXPECT_THROW(processor.execute(img, 1.0f, 1u), std::runtime_error);

	processor = LUT3DPipelineGPU(&lut3d, &calls);
	EXPECT_NO_THROW(processor.execute(img, 1.0f, 1u));
}
#endif