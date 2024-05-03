#include <ImageProcessing/ImageProcessor.hpp>
#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <fmt/format.h>

ImageProcessor::ImageProcessor(const DataLoader &ld) : loader(ld) {}

void ImageProcessor::save() const
{
    const auto& outputPath = loader.getInputParams().getOutputImgPath();
    std::cout << fmt::format("[INFO] Saving image to: {}\n", outputPath);
    try
    {
        cv::imwrite(outputPath, newImg);
    }
    catch (cv::Exception &ex)
    {
        std::cerr << fmt::format("[ERROR] {}\n", ex.what());
    }
}

void ImageProcessor::execute()
{
    process();
    save();
}
