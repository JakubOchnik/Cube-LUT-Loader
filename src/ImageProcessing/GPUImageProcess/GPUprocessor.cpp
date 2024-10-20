#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/GPUImageProcess/GPUprocessor.hpp>
#include <ImageProcessing/GPUImageProcess/LUT3D/applyNearestValueHost.hpp>
#include <ImageProcessing/GPUImageProcess/LUT3D/applyTrilinearHost.hpp>
#include <ImageProcessing/GPUImageProcess/Utils/CudaUtils.hpp>
#include <iostream>

GpuProcessor::GpuProcessor(FileIO& fileIfc) : ImageProcessor(fileIfc) {}

cv::Mat GpuProcessor::process(float strength, InterpolationMethod method)
{
	// TODO: Implement as standalone commands retrieved from a map
	// with keys equal to command line args. Use inheritance to respect
	// DRY (there are tons of similar code in the different interpolation classes).
	std::cout << "[INFO] Processing the image...\n";
	if (fileInterface.getCube().getType() != LUTType::LUT3D) {
		// cout << "Applying basic 1D LUT..." << endl;
		throw std::runtime_error("GPU-accelerated 1D LUTs are not implemented yet");
		// newImg = applyBasic1D(loader.getImg(), loader.getCube(), opacity);
	} else if (method == InterpolationMethod::Trilinear) {
		std::cout << "[INFO] Applying trilinear interpolation...\n";
		newImg = GpuTrilinear::applyTrilinearGpu(
			fileInterface.getImg(), std::get<Table3D>(fileInterface.getCube().getTable()), strength, threadsPerBlock);
	} else {
		std::cout << "[INFO] Applying nearest-value interpolation...\n";
		newImg = GpuNearestVal::applyNearestGpu(
			fileInterface.getImg(), std::get<Table3D>(fileInterface.getCube().getTable()), strength, threadsPerBlock);
	}
	return newImg;
}

GpuProcessor::~GpuProcessor()
{
	if (newImg.data)
	{
		CudaUtils::freeUnifiedPtr<unsigned char>(newImg.data);
	}
}
