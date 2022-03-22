#include <ImageProcess/Processor.hpp>

Processor::Processor(const Loader& ld) : loader(ld)
{
}

cv::Mat_<cv::Vec3b> Processor::process()
{
	std::cout << "Processing the image...\n";
	if (const float opacity = loader.getVm()["strength"].as<float>(); !loader.getCube().is3D())
	{
		cout << "Applying basic 1D LUT..." << endl;
		newImg = Basic1D::applyBasic1D(loader.getImg(), loader.getCube(), opacity, loader.getThreads());
	}
	else if (loader.getVm().count("trilinear"))
	{
		cout << "Applying trilinear interpolation..." << endl;
		newImg = Trilinear::applyTrilinear(loader.getImg(), loader.getCube(), opacity, loader.getThreads());
	}
	else
	{
		cout << "Applying nearest-value interpolation..." << endl;
		newImg = NearestValue::applyNearest(loader.getImg(), loader.getCube(), opacity, loader.getThreads());
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
