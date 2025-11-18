#include "ImageProcessing/OpenCL/LUT3DOpenCLImpl.hpp"
#include "FileIO/CubeLUT.hpp"
#include "TaskDispatcher/InputParams.h"
#include <CL/cl.h>
#include <cstring>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <stdexcept>
#include <vector>

LUT3DOpenCLImpl::LUT3DOpenCLImpl(Table3D* lut, OpenCLRuntime* OCLRuntime,
										 InterpolationMethod interpolationMethod)
	: lut3d(lut), runtime(OCLRuntime), method(interpolationMethod) {}

cl_image_format LUT3DOpenCLImpl::getLUTFormat() const {
	cl_image_format format;
	format.image_channel_order = CL_RGBA; // store LUT as RGBA
	format.image_channel_data_type = CL_FLOAT;
	return format;
}

std::vector<float> LUT3DOpenCLImpl::prepareLUTImage() const {
	const int lutSize = lut3d->dimension(0);
	const int channels = lut3d->dimension(3); // usually 3 (RGB)
	if (channels < 3 || channels > 4) {
		throw std::runtime_error("LUT must have 3 or 4 channels");
	}

	std::vector<float> lutImage(lutSize * lutSize * lutSize * 4, 0.0f);
	for (int b = 0; b < lutSize; ++b)
		for (int g = 0; g < lutSize; ++g)
			for (int r = 0; r < lutSize; ++r) {
				int idx4 = ((b * lutSize + g) * lutSize + r) * 4;
				lutImage[idx4 + 0] = (*lut3d)(r, g, b, 0); // R
				lutImage[idx4 + 1] = (*lut3d)(r, g, b, 1); // G
				lutImage[idx4 + 2] = (*lut3d)(r, g, b, 2); // B
				if (channels == 4) {
					lutImage[idx4 + 3] = (*lut3d)(r, g, b, 3); // A
				} else {
					lutImage[idx4 + 3] = 0.0f; // unused
				}
			}

	return lutImage;
}

cv::Mat LUT3DOpenCLImpl::execute(cv::Mat img, const float opacity, const uint threadPool) {

	if (!lut3d) {
		throw std::runtime_error("3D LUT is unavailable");
	}

	if (!runtime) {
		throw std::runtime_error("OpenCL runtime is unavailable");
	}

	int width{img.cols}, height{img.rows};
	const int channels{3};
	const int imgSize = width * height * channels * sizeof(unsigned char);
	const int lutSize = static_cast<int>(pow(lut3d->dimension(0), 3) * channels * sizeof(float));
  const int N = lut3d->dimension(0);

	std::vector<float> lutImage = prepareLUTImage();
	cl_image_format format = getLUTFormat();

	cl_image_desc desc = {};
	desc.image_type = CL_MEM_OBJECT_IMAGE3D;
	desc.image_width = N;
	desc.image_height = N;
	desc.image_depth = N;

	cl_mem lut3dImage = clCreateImage(runtime->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &desc,
									  lutImage.data(), nullptr);

	unsigned char* imgPtr = new unsigned char[imgSize];

	memcpy(imgPtr, img.data, imgSize);

	std::cout << "Creating buffers" << std::endl;
	cl_mem imageBuffer = runtime->createBuffer(imgSize, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imgPtr);

	std::cout << "Creating kkernel" << std::endl;

	char lutDimension = static_cast<char>(lut3d->dimension(0));

	cl_kernel kernel;

	switch (method) {
	case InterpolationMethod::Trilinear: {
		kernel = runtime->createKernel(
			"/home/alubinski/Work/Cube-LUT-Loader/src/ImageProcessing/OpenCL/kernels/linear3DValue.cl",
			"applyLinear3DKernel");
		break;
	}
	case InterpolationMethod::Tetrahedral: {
		std::runtime_error("Tetrahedral interpolation unavailable in OpenCL implementation");
	}
	case InterpolationMethod::NearestValue:
	default:
		kernel = runtime->createKernel(
			"/home/alubinski/Work/Cube-LUT-Loader/src/ImageProcessing/OpenCL/kernels/nearestVelue.cl",
			"applyNearestKernel");
	}

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
	clSetKernelArg(kernel, 1, sizeof(int), &channels);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &lut3dImage);
	clSetKernelArg(kernel, 3, sizeof(float), &opacity);
	clSetKernelArg(kernel, 4, sizeof(int), &width);
	clSetKernelArg(kernel, 5, sizeof(int), &height);

	size_t globalWorkSize[2] = {(size_t)width, (size_t)height};
  size_t localWorkSize[2] {threadPool, threadPool};
	clEnqueueNDRangeKernel(runtime->queue(), kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr,
						   nullptr);
	clFinish(runtime->queue());

	auto finalImg = cv::Mat(height, width, CV_8UC3);
	clEnqueueReadBuffer(runtime->queue(), imageBuffer, CL_TRUE, 0, imgSize, finalImg.data, 0, nullptr, nullptr);

	delete[] imgPtr;
	return finalImg;
}
