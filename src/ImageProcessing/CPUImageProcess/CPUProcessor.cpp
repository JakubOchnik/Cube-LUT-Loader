#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/CPUImageProcess/LUT1D/apply1D.hpp>
#include <ImageProcessing/CPUImageProcess/LUT3D/applyNearestValue.hpp>
#include <ImageProcessing/CPUImageProcess/LUT3D/applyTrilinear.hpp>
#include <ImageProcessing/CPUImageProcess/CPUProcessor.hpp>
#include <iostream>

CPUProcessor::CPUProcessor(FileIO& fileIfc, uint threads) : ImageProcessor(fileIfc), numberOfThreads(threads) {}

cv::Mat CPUProcessor::process(float strength, InterpolationMethod method)
{
	// TODO: Implement as standalone commands retrieved from a map
	// with keys equal to command line args. Use inheritance to respect
	// DRY (there are tons of similar code in the different interpolation classes).
	std::cout << "[INFO] Processing the image...\n";
	if (fileInterface.getCube().getType() != LUTType::LUT3D)
	{
		std::cout << "[INFO] Applying basic 1D LUT...\n";
		newImg = Basic1D::applyBasic1D(fileInterface.getImg(), std::get<Table1D>(fileInterface.getCube().getTable()),
									   strength, numberOfThreads);
	}
	else if (method == InterpolationMethod::Trilinear)
	{
		std::cout << "[INFO] Applying trilinear interpolation...\n";
		newImg = Trilinear::applyTrilinear(fileInterface.getImg(), std::get<Table3D>(fileInterface.getCube().getTable()),
										   strength, numberOfThreads);
	}
	else
	{
		std::cout << "[INFO] Applying nearest-value interpolation...\n";
		newImg = NearestValue::applyNearest(fileInterface.getImg(), std::get<Table3D>(fileInterface.getCube().getTable()),
											strength, numberOfThreads);
	}
	return newImg;
}
