#include <ImageProcessing/ImageProcessor.hpp>
#include <boost/program_options.hpp>

ImageProcessor::ImageProcessor(const DataLoader &ld) : loader(ld) {}

void ImageProcessor::save() const
{
    std::cout << "Saving...\n";
    const auto name = loader.getVm()["output"].as<std::string>();
    try
    {
        imwrite(name, newImg);
    }
    catch (cv::Exception &e)
    {
        std::cerr << e.what() << "\n"; // output exception message
    }
}

void ImageProcessor::execute()
{
    process();
    save();
}
