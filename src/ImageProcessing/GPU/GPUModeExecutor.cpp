#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/GPU/GPUModeExecutor.hpp>
#include <ImageProcessing/GPU/NearestValImplGPU.hpp>
#include <ImageProcessing/GPU/TrilinearImplGPU.hpp>
#include <ImageProcessing/GPU/Utils/CudaUtils.hpp>
#include <iostream>

GPUModeExecutor::GPUModeExecutor(FileIO& fileIfc) : ImageProcessExecutor(fileIfc) {}

cv::Mat GPUModeExecutor::process(float intensity, InterpolationMethod method)
{
	std::cout << "[INFO] Processing the image...\n";
	if (fileInterface.getCube().getType() != LUTType::LUT3D) {
		throw std::runtime_error("GPU-accelerated 1D LUTs are not implemented yet");
	} else if (method == InterpolationMethod::Trilinear) {
		std::cout << "[INFO] Applying trilinear interpolation...\n";
		auto lut = std::get<Table3D>(fileInterface.getCube().getTable());
		newImg = TrilinearImplGPU(&lut, &calls).execute(fileInterface.getImg(), intensity, threadsPerBlock);
	} else {
		std::cout << "[INFO] Applying nearest-value interpolation...\n";
		auto lut = std::get<Table3D>(fileInterface.getCube().getTable());
		newImg = NearestValImplGPU(&lut, &calls).execute(fileInterface.getImg(), intensity, threadsPerBlock);
	}
	return newImg;
}

GPUModeExecutor::~GPUModeExecutor()
{
	if (newImg.data)
	{
		CudaUtils::freeUnifiedPtr<unsigned char>(newImg.data, calls);
	}
}
