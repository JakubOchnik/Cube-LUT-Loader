#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/CPU/CPUModeExecutor.hpp>
#include <ImageProcessing/CPU/NearestValImplCPU.hpp>
#include <ImageProcessing/CPU/Simple1DImplCPU.hpp>
#include <ImageProcessing/CPU/TrilinearImplCPU.hpp>
#include <ImageProcessing/CPU/TetrahedralImplCPU.hpp>
#include <iostream>

CPUModeExecutor::CPUModeExecutor(FileIO& fileIfc, uint threads)
	: ImageProcessExecutor(fileIfc), numberOfThreads(threads) {}

cv::Mat CPUModeExecutor::process(float intensity, InterpolationMethod method) {
	std::cout << "[INFO] Processing the image...\n";
	if (fileInterface.getCube().getType() != LUTType::LUT3D) {
		std::cout << "[INFO] Applying basic 1D LUT...\n";
		auto lut = std::get<Table1D>(fileInterface.getCube().getTable());
		newImg = Simple1DImplCPU(&lut).execute(fileInterface.getImg(), intensity, numberOfThreads);
	} else if (method == InterpolationMethod::Trilinear) {
		std::cout << "[INFO] Applying trilinear interpolation...\n";
		auto lut = std::get<Table3D>(fileInterface.getCube().getTable());
		newImg = TrilinearImplCPU(&lut).execute(fileInterface.getImg(), intensity, numberOfThreads);
	}
	else if (method == InterpolationMethod::Tetrahedral) {
		std::cout << "[INFO] Applying tetrahedral interpolation...\n";
		auto lut = std::get<Table3D>(fileInterface.getCube().getTable());
		newImg = TetrahedralImplCPU(&lut).execute(fileInterface.getImg(), intensity, numberOfThreads);
	} else {
		std::cout << "[INFO] Applying nearest-value interpolation...\n";
		auto lut = std::get<Table3D>(fileInterface.getCube().getTable());
		newImg = NearestValImplCPU(&lut).execute(fileInterface.getImg(), intensity, numberOfThreads);
	}
	return newImg;
}
