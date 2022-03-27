#include <GPUImageProcess/GPUprocessor.hpp>

GpuProcessor::GpuProcessor(const DataLoader& ld) : loader(ld)
{
}

cv::Mat GpuProcessor::process()
{
	// TODO: Implement as standalone commands retrieved from a map
	// with keys equal to command line args. Use inheritance to respect
	// DRY (there are tons of similar code in the different interpolation classes).
	// TODO: Catch exceptions (after implementing CUDA error handling)
	std::cout << "Processing the image...\n";
	if (const float opacity = loader.getVm()["strength"].as<float>(); !loader.getCube().is3D())
	{
		// cout << "Applying basic 1D LUT..." << endl;
		cout << "GPU-accelerated 1D LUTs are not implemented yet" << endl;
		//newImg = applyBasic1D(loader.getImg(), loader.getCube(), opacity);
	}
	else if (loader.getVm().count("trilinear"))
	{
		cout << "Applying trilinear interpolation..." << endl;
		newImg = GpuTrilinear::applyTrilinearGpu(loader, opacity, threadsPerBlock);
	}
	else
	{
		cout << "Applying nearest-value interpolation..." << endl;
		newImg = GpuNearestVal::applyNearestGpu(loader, opacity, threadsPerBlock);
	}
	return newImg;
}

void GpuProcessor::save() const
{
	std::cout << "Saving...\n";
	const std::string name{loader.getVm()["output"].as<std::string>()};
	try
	{
		imwrite(name, newImg);
	}
	catch (cv::Exception& e)
	{
		cerr << e.what() << endl; // output exception message
	}
	CudaUtils::freeUnifiedPtr<unsigned char>(newImg.data);
}

void GpuProcessor::execute()
{
	process();
	save();
}
