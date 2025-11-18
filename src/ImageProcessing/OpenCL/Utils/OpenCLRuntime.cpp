#include "ImageProcessing/OpenCL/Utils/OpenRuntime.hpp"
#include "fstream"
#include <CL/cl.h>
#include <CL/cl_platform.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

void OpenCLRuntime::initialize() {
	cl_int result{CL_SUCCESS};
	cl_uint numberOfPlatforms{0};

	result = clGetPlatformIDs(0, nullptr, &numberOfPlatforms);
	if (result != CL_SUCCESS) {
		throw std::runtime_error("Failed to get number of avaible OpenCL platforms");
	}

	std::vector<cl_platform_id> platforms(numberOfPlatforms);
	result = clGetPlatformIDs(numberOfPlatforms, platforms.data(), nullptr);
	if (result != CL_SUCCESS) {
		throw std::runtime_error("Failed to get OpenCL platforms ids");
	}

	platform_ = platforms[0];

	cl_uint numberOfDevices{0};
	result = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, nullptr, &numberOfDevices);
	if (result != CL_SUCCESS) {
		throw std::runtime_error("Failed to get number of avaible OpenCL devices");
	}

	std::vector<cl_device_id> devices(numberOfDevices);
	result = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, numberOfDevices, devices.data(), nullptr);
	if (result != CL_SUCCESS) {
		throw std::runtime_error("Failed to get OpenCL device ids");
	}

	device_ = devices[0];
	context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &result);
	if (result != CL_SUCCESS) {
		throw std::runtime_error("Failed to create OpenCL context");
	}

	queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &result);
	if (result != CL_SUCCESS) {
		throw std::runtime_error("Failed to create OpenCL command queue");
	}
}

cl_mem OpenCLRuntime::createBuffer(const size_t size, const cl_mem_flags flags, void* hostPtr) {
	cl_int result{CL_SUCCESS};

	cl_mem buffer = clCreateBuffer(context_, flags, size, hostPtr, &result);

	if (result != CL_SUCCESS) {
		throw std::runtime_error("Failed to create OpenCL buffer");
	}
	return buffer;
}

cl_kernel OpenCLRuntime::createKernel(const std::string& filename, const std::string& kernelName) {
	// 1. Read kernel source from file
	std::ifstream file(filename);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open kernel file: " + filename);
	}
	std::stringstream ss;
	ss << file.rdbuf();
	std::string kernelSourceStr = ss.str();
	const char* kernelSource = kernelSourceStr.c_str();
	size_t kernelSize = kernelSourceStr.size();

	cl_int err;

	// 2. Create program from source
	cl_program program = clCreateProgramWithSource(context_, 1, &kernelSource, &kernelSize, &err);
	if (err != CL_SUCCESS) {
		throw std::runtime_error("Failed to create OpenCL program from source: " + std::to_string(err));
	}

	// 3. Build the program
	err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
	if (err != CL_SUCCESS) {
		// If build failed, get build log
		size_t logSize;
		clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
		std::vector<char> log(logSize);
		clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
		std::cerr << "OpenCL Build Log:\n" << log.data() << std::endl;

		clReleaseProgram(program);
		throw std::runtime_error("Failed to build OpenCL program.");
	}

	// 4. Create kernel
	cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
	if (err != CL_SUCCESS) {
		clReleaseProgram(program);
		throw std::runtime_error("Failed to create kernel: " + std::to_string(err));
	}

	// Note: program object can be released after kernel creation
	clReleaseProgram(program);

	return kernel;
}
