#include <ImageProcessing/ImageProcessExecutor.hpp>
#include <fmt/format.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

ImageProcessExecutor::ImageProcessExecutor(FileIO& fileIfc) : fileInterface(fileIfc) {}

cv::Mat ImageProcessExecutor::execute(float strength, cv::Size dstImageSize, InterpolationMethod method) {
	// Swap original image with the resized one to avoid storing two images simultaneously in memory
	fileInterface.setImg(resizeImage(fileInterface.getImg(), dstImageSize));
	return process(strength, method);
}

cv::Mat ImageProcessExecutor::resizeImage(cv::Mat inputImg, cv::Size size, int interpolationMode) {
	const auto [width, height] = size;
	if (!width && !height) {
		return inputImg;
	}
	size.width = width ? width : inputImg.size().width;
	size.height = height ? height : inputImg.size().height;
	std::cout << fmt::format("[INFO] Scaling image to {}x{}\n", size.width, size.height);
	cv::Mat newImage;
	cv::resize(inputImg, newImage, size, 0, 0, interpolationMode);
	return newImage;
}
