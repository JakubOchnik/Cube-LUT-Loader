#include "GPUprocessor.hpp"

GpuProcessor::GpuProcessor(const Loader& ld) : loader(ld)
{
}

bool GpuProcessor::is3D() const
{
	return loader.getCube().LUT1D.empty();
}

cv::Mat GpuProcessor::process()
{
	std::cout << "Processing the image...\n";
	if (const float opacity = loader.getVm()["strength"].as<float>(); !is3D())
	{
		// cout << "Applying basic 1D LUT..." << endl;
		cout << "GPU-accelerated 1D LUTs are not implemented yet" << endl;
		//newImg = applyBasic1D(loader.getImg(), loader.getCube(), opacity);
	}
	else if (loader.getVm().count("trilinear"))
	{
		cout << "Applying trilinear interpolation..." << endl;
		newImg = applyTrilinearGpu(loader, opacity);
	}
	else
	{
		cout << "Applying nearest-value interpolation..." << endl;
		newImg = applyNearestGpu(loader, opacity);
	}
	return newImg;
}

void GpuProcessor::save() const
{
	std::cout << "Saving...\n";
	std::string name = loader.getVm()["output"].as<std::string>();
	try {
		imwrite(name, newImg);
	}
	catch (cv::Exception& e)
	{
		cerr << e.what() << endl; // output exception message
	}
	CudaUtils::freeUnifiedPtr<unsigned char>(newImg.data);
}

void GpuProcessor::perform()
{
	process();
	save();
}
