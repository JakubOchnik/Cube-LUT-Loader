#include <DataLoader/CubeLUT.hpp>
#include <ImageProcessing/GPUImageProcess/GPUprocessor.hpp>
#include <ImageProcessing/GPUImageProcess/LUT3D/applyNearestValueHost.hpp>
#include <ImageProcessing/GPUImageProcess/LUT3D/applyTrilinearHost.hpp>
#include <ImageProcessing/GPUImageProcess/Utils/CudaUtils.hpp>
#include <boost/program_options.hpp>

GpuProcessor::GpuProcessor(const DataLoader &ld) : ImageProcessor(ld) {}

cv::Mat GpuProcessor::process()
{
	// TODO: Implement as standalone commands retrieved from a map
	// with keys equal to command line args. Use inheritance to respect
	// DRY (there are tons of similar code in the different interpolation classes).
	std::cout << "Processing the image...\n";
	if (const float opacity = loader.getVm()["strength"].as<float>();
		!loader.getCube().is3D())
	{
		// cout << "Applying basic 1D LUT..." << endl;
		std::cout << "GPU-accelerated 1D LUTs are not implemented yet\n";
		// newImg = applyBasic1D(loader.getImg(), loader.getCube(), opacity);
	}
	else if (loader.getVm().count("trilinear"))
	{
		std::cout << "Applying trilinear interpolation...\n";
		newImg =
			GpuTrilinear::applyTrilinearGpu(loader, opacity, threadsPerBlock);
	}
	else
	{
		std::cout << "Applying nearest-value interpolation...\n";
		newImg =
			GpuNearestVal::applyNearestGpu(loader, opacity, threadsPerBlock);
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
