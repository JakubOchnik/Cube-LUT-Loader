#include <ImageProcessing/ImageProcessor.hpp>
#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <format>

ImageProcessor::ImageProcessor(const DataLoader &ld) : loader(ld) {}

void ImageProcessor::save() const
{
    const auto& outputPath = loader.getInputParams().getOutputImgPath();
    std::cout << std::format("[INFO] Saving image to: {}\n", outputPath);
    try
    {
        cv::imwrite(outputPath, newImg);
    }
    catch (cv::Exception &ex)
    {
        std::cerr << std::format("[ERROR] {}\n", ex.what());
    }
}

void ImageProcessor::execute()
{
    process();
    save();
}
