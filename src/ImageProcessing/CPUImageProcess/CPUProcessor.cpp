#include <DataLoader/CubeLUT.hpp>
#include <ImageProcessing/CPUImageProcess/LUT1D/apply1D.hpp>
#include <ImageProcessing/CPUImageProcess/LUT3D/applyNearestValue.hpp>
#include <ImageProcessing/CPUImageProcess/LUT3D/applyTrilinear.hpp>
#include <ImageProcessing/CPUImageProcess/CPUProcessor.hpp>
#include <boost/program_options.hpp>
#include <iostream>

CPUProcessor::CPUProcessor(const DataLoader &ld) : ImageProcessor(ld) {}

cv::Mat CPUProcessor::process()
{
	// TODO: Implement as standalone commands retrieved from a map
	// with keys equal to command line args. Use inheritance to respect
	// DRY (there are tons of similar code in the different interpolation classes).
	std::cout << "Processing the image...\n";
	if (const float opacity = loader.getVm()["strength"].as<float>();
		!loader.getCube().is3D())
	{
		std::cout << "Applying basic 1D LUT...\n";
		newImg = Basic1D::applyBasic1D(loader.getImg(), loader.getCube(),
									   opacity, loader.getThreads());
	}
	else if (loader.getVm().count("trilinear"))
	{
		std::cout << "Applying trilinear interpolation...\n";
		newImg = Trilinear::applyTrilinear(loader.getImg(), loader.getCube(),
										   opacity, loader.getThreads());
	}
	else
	{
		std::cout << "Applying nearest-value interpolation...\n";
		newImg = NearestValue::applyNearest(loader.getImg(), loader.getCube(),
											opacity, loader.getThreads());
	}
	return newImg;
}
