#include <ImageProcessing/ImageProcessor.hpp>
#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <boost/format.hpp>

ImageProcessor::ImageProcessor(const DataLoader &ld) : loader(ld) {}

void ImageProcessor::save() const
{
    const auto& outputPath = loader.getInputParams().getOutputImgPath();
    std::cout << boost::format("[INFO] Saving image to: %1%\n") % outputPath;
    try
    {
        cv::imwrite(outputPath, newImg);
    }
    catch (cv::Exception &ex)
    {
        std::cerr << boost::format("[ERROR] %1%\n") % ex.what();
    }
}

void ImageProcessor::execute()
{
    process();
    save();
}
