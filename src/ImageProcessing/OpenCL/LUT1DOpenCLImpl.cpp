#include "ImageProcessing/OpenCL/LUT1DOpenCLImpl.hpp"
#include "FileIO/CubeLUT.hpp"
#include "TaskDispatcher/InputParams.h"
#include <CL/cl.h>
#include <cstring>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <stdexcept>
#include <vector>

LUT1DImplOpenCL::LUT1DImplOpenCL(Table1D* lut, OpenCLRuntime* OCLRuntime)
	: lut1d(lut), runtime(OCLRuntime) {}

cl_image_format LUT1DImplOpenCL::getLUTFormat() const {
	cl_image_format format;
	format.image_channel_order = CL_RGBA; // store LUT as RGBA
	format.image_channel_data_type = CL_FLOAT;
	return format;
}

std::vector<float> LUT1DImplOpenCL::prepareLUTImage() const {
  const int lutSize   = lut1d->dimension(0);
    const int channels  = lut1d->dimension(1);   // 3 or 4

    if (channels < 3 || channels > 4) {
        throw std::runtime_error("1D LUT must have 3 or 4 channels");
    }

    // OpenCL 1D image always needs 4 channels (RGBA) in float
    std::vector<float> lutImage(lutSize * 4, 0.0f);

    for (int i = 0; i < lutSize; ++i)
    {
        int idx = i * 4;

        lutImage[idx + 0] = (*lut1d)(i, 0); // R
        lutImage[idx + 1] = (*lut1d)(i, 1); // G
        lutImage[idx + 2] = (*lut1d)(i, 2); // B

        if (channels == 4)
            lutImage[idx + 3] = (*lut1d)(i, 3); // A
        else
            lutImage[idx + 3] = 0.0f;           // pad
    }

    return lutImage;
}

cv::Mat LUT1DImplOpenCL::execute(cv::Mat img, const float opacity, const uint threadPool) {

	if (!lut1d) {
		throw std::runtime_error("3D LUT is unavailable");
	}

	if (!runtime) {
		throw std::runtime_error("OpenCL runtime is unavailable");
	}

	int width{img.cols}, height{img.rows};
	const int channels{3};
	const int imgSize = width * height * channels * sizeof(unsigned char);
  const int N = lut1d->dimension(0);

	std::vector<float> lutImage = prepareLUTImage();
	cl_image_format format = getLUTFormat();

	cl_image_desc desc = {};
	desc.image_type = CL_MEM_OBJECT_IMAGE1D;
	desc.image_width = N;

	cl_mem lut1dImage = clCreateImage(runtime->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &desc,
									  lutImage.data(), nullptr);

	unsigned char* imgPtr = new unsigned char[imgSize];

	memcpy(imgPtr, img.data, imgSize);

	std::cout << "Creating buffers" << std::endl;
	cl_mem imageBuffer = runtime->createBuffer(imgSize, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, imgPtr);

	std::cout << "Creating kkernel" << std::endl;

	cl_kernel kernel;

  kernel = runtime->createKernel(
      "/home/alubinski/Work/Cube-LUT-Loader/src/ImageProcessing/OpenCL/kernels/linear1DValue.cl",
      "applyLinear1DKernel");

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &imageBuffer);
	clSetKernelArg(kernel, 1, sizeof(int), &channels);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &lut1dImage);
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
