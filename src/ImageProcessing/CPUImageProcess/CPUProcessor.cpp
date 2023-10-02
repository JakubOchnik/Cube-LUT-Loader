#include <DataLoader/CubeLUT.hpp>
#include <ImageProcessing/CPUImageProcess/LUT1D/apply1D.hpp>
#include <ImageProcessing/CPUImageProcess/LUT3D/applyNearestValue.hpp>
#include <ImageProcessing/CPUImageProcess/LUT3D/applyTrilinear.hpp>
#include <ImageProcessing/CPUImageProcess/CPUProcessor.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <chrono>

CPUProcessor::CPUProcessor(const DataLoader &ld) : ImageProcessor(ld) {}

cv::Mat CPUProcessor::process()
{
	// TODO: Implement as standalone commands retrieved from a map
	// with keys equal to command line args. Use inheritance to respect
	// DRY (there are tons of similar code in the different interpolation classes).
	std::cout << "[INFO] Processing the image...\n";
	if (const float opacity = loader.getVm()["strength"].as<float>(); loader.getCube().getType() != LUTType::LUT3D)
	{
		std::cout << "[INFO] Applying basic 1D LUT...\n";
		newImg = Basic1D::applyBasic1D(loader.getImg(), std::get<Table1D>(loader.getCube().getTable()),
									   opacity, loader.getThreads());
	}
	else if (loader.getVm().count("trilinear"))
	{
		std::cout << "[INFO] Applying trilinear interpolation...\n";

		const auto start = std::chrono::steady_clock::now();
		newImg = Trilinear::applyTrilinear(loader.getImg(), std::get<Table3D>(loader.getCube().getTable()),
										   opacity, loader.getThreads());
		const auto end = std::chrono::steady_clock::now();
		std::cout << "Execution took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms";
	}
	else
	{
		std::cout << "[INFO] Applying nearest-value interpolation...\n";
		const auto start = std::chrono::steady_clock::now();
		newImg = NearestValue::applyNearest(loader.getImg(), std::get<Table3D>(loader.getCube().getTable()),
											opacity, loader.getThreads());
		const auto end = std::chrono::steady_clock::now();
		std::cout << "Execution took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms";											
	}
	return newImg;
}
