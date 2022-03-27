#include <ImageProcess/Processor.hpp>

Processor::Processor(const DataLoader& ld) : loader(ld)
{
}

cv::Mat_<cv::Vec3b> Processor::process()
{
	// TODO: Implement as standalone commands retrieved from a map
	// with keys equal to command line args. Use inheritance to respect
	// DRY (there are tons of similar code in the different interpolation classes).
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

void Processor::execute()
{
	process();
	save();
}
