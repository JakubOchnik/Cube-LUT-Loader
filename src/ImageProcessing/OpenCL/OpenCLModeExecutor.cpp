
#include "ImageProcessing/OpenCL/OpenCLModeExecutor.hpp"
#include "ImageProcessing/OpenCL/LUT1DOpenCLImpl.hpp"
#include "ImageProcessing/OpenCL/LUT3DOpenCLImpl.hpp"
#include "TaskDispatcher/InputParams.h"
#include <FileIO/CubeLUT.hpp>
#include <iostream>

OpenCLModeExecutor::OpenCLModeExecutor(FileIO& fileIfc) : ImageProcessExecutor(fileIfc) {}

cv::Mat OpenCLModeExecutor::process(float intensity, InterpolationMethod method) {
	std::cout << "[INFO] Processing the image...\n";
	if (fileInterface.getCube().getType() != LUTType::LUT3D) {
    std::cout << "[INFO] Applying linear interpolation...\n";
		auto lut = std::get<Table1D>(fileInterface.getCube().getTable());
		newImg = LUT1DImplOpenCL(&lut, &runtime).execute(fileInterface.getImg(), intensity, workGroups);
	} else if (method == InterpolationMethod::Trilinear) {
		std::cout << "[INFO] Applying trilinear interpolation...\n";
		auto lut = std::get<Table3D>(fileInterface.getCube().getTable());
		newImg = LUT3DOpenCLImpl(&lut, &runtime, InterpolationMethod::Trilinear)
					 .execute(fileInterface.getImg(), intensity, workGroups);
	} else if (method == InterpolationMethod::Tetrahedral) {
		throw std::runtime_error("GPU-accelerated tetrahedral interpolation is not implemented yet");
	} else {
		std::cout << "[INFO] Applying nearest-value interpolation...\n";
		auto lut = std::get<Table3D>(fileInterface.getCube().getTable());
		newImg = LUT3DOpenCLImpl(&lut, &runtime, InterpolationMethod::NearestValue)
					 .execute(fileInterface.getImg(), intensity, workGroups);
	}
	return newImg;
}
