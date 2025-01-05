#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ImageProcessing/GPU/Utils/CudaCalls.hpp>
#include <gmock/gmock.h>

class CudaCallsMock : public CudaCalls {
public:
	MOCK_METHOD(cudaError_t, cudaMalloc, (void**, size_t), (override, const));
	MOCK_METHOD(cudaError_t, cudaMemcpy, (void*, const void*, size_t, enum cudaMemcpyKind), (override, const));
	MOCK_METHOD(cudaError_t, cudaMallocManaged, (unsigned char**, size_t, unsigned int), (override, const));
	MOCK_METHOD(cudaError_t, cudaFree, (void*), (override, const));
};
