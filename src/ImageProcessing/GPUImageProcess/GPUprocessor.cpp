#include <FileIO/CubeLUT.hpp>
#include <ImageProcessing/GPUImageProcess/GPUprocessor.hpp>
#include <ImageProcessing/GPUImageProcess/LUT3D/applyNearestValueHost.hpp>
#include <ImageProcessing/GPUImageProcess/LUT3D/applyTrilinearHost.hpp>
#include <ImageProcessing/GPUImageProcess/Utils/CudaUtils.hpp>
#include <boost/program_options.hpp>
#include <iostream>

GpuProcessor::GpuProcessor(const FileIO &ld) : ImageProcessor(ld) {}

cv::Mat GpuProcessor::process()
{
	// TODO: Implement as standalone commands retrieved from a map
	// with keys equal to command line args. Use inheritance to respect
	// DRY (there are tons of similar code in the different interpolation classes).
	auto& inputImg = loader.getImg();
	std::cout << "[INFO] Processing the image...\n";
	const auto& inputParams = loader.getInputParams();
	if (const float opacity = inputParams.getEffectStrength(); loader.getCube().getType() != LUTType::LUT3D)
	{
		// cout << "Applying basic 1D LUT..." << endl;
		throw std::runtime_error("GPU-accelerated 1D LUTs are not implemented yet");
		// newImg = applyBasic1D(loader.getImg(), loader.getCube(), opacity);
	}
	else if (inputParams.getInterpolationMethod() == InterpolationMethod::Trilinear)
	{
		std::cout << "[INFO] Applying trilinear interpolation...\n";
		newImg = GpuTrilinear::applyTrilinearGpu(inputImg, std::get<Table3D>(loader.getCube().getTable()), opacity, threadsPerBlock);
	}
	else
	{
		std::cout << "[INFO] Applying nearest-value interpolation...\n";
		newImg = GpuNearestVal::applyNearestGpu(inputImg, std::get<Table3D>(loader.getCube().getTable()), opacity, threadsPerBlock);
	}
	return newImg;
}

void GpuProcessor::execute()
{
	if (!isCudaAvailable())
	{
		return;
	}
	ImageProcessor::execute();
}

bool GpuProcessor::isCudaAvailable() const
{
	if (!CudaUtils::isCudaDriverAvailable())
	{
		std::cerr << "ERROR (CUDA): CUDA driver was not detected\n";
	}
	if (!CudaUtils::isCudaDeviceAvailable())
	{
		return false;
	}
	return true;
}

GpuProcessor::~GpuProcessor()
{
	if (newImg.data)
	{
		CudaUtils::freeUnifiedPtr<unsigned char>(newImg.data);
	}
}
