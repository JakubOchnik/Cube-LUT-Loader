#include <ImageProcessing/ImageProcessExecutor.hpp>
#include <fmt/format.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

ImageProcessExecutor::ImageProcessExecutor(FileIO& fileIfc) : fileInterface(fileIfc) {}

cv::Mat ImageProcessExecutor::execute(float intensity, cv::Size dstImageSize, InterpolationMethod method) {

	resizeImage(fileInterface.getImg(), dstImageSize, fileInterface.getAlpha());
	return process(intensity, method);
}

void ImageProcessExecutor::resizeImage(cv::Mat3b& inputImg, cv::Size size, cv::Mat1b& inputAlpha, int interpolationMode) {
	const auto [width, height] = size;
	if (!width && !height) {
		return;
	}
	size.width = width ? width : inputImg.size().width;
	size.height = height ? height : inputImg.size().height;
	std::cout << fmt::format("[INFO] Scaling image to {}x{}\n", size.width, size.height);
	cv::resize(inputImg, inputImg, size, 0, 0, interpolationMode);
	if (!inputAlpha.empty()) {
		cv::resize(inputAlpha, inputAlpha, size, 0, 0, interpolationMode);
	}
}
