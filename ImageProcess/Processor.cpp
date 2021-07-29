#include "Processor.hpp"

Processor::Processor(const Loader& ld) : loader(ld)
{
}

bool Processor::is3D() const
{
	return loader.getCube().LUT1D.empty();
}

cv::Mat_<cv::Vec3b> Processor::process()
{
	std::cout << "Processing the image...\n";
	if (const float opacity = loader.getVm()["strength"].as<float>(); !is3D())
	{
		cout << "Applying basic 1D LUT..." << endl;
		newImg = applyBasic1D(loader.getImg(), loader.getCube(), opacity);
	}
	else if (loader.getVm().count("trilinear"))
	{
		cout << "Applying trilinear interpolation..." << endl;
		newImg = applyTrilinear(loader.getImg(), loader.getCube(), opacity);
	}
	else
	{
		cout << "Applying nearest-value interpolation..." << endl;
		newImg = applyNearest(loader.getImg(), loader.getCube(), opacity);
	}
	return newImg;
}

void Processor::save() const
{
	std::cout << "Saving...\n";
	imwrite(loader.getVm()["output"].as<std::string>(), newImg);
}

void Processor::perform()
{
	process();
	save();
}
