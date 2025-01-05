#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaCalls {
public:
	virtual cudaError_t cudaMalloc(void** p, size_t size) const { return ::cudaMalloc(p, size); }

	virtual cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) const {
		return ::cudaMemcpy(dst, src, count, kind);
	}

	// cudaMallocManaged is a template function, however we only need unsigned char for now
	// and it's hard to mock template functions
	virtual cudaError_t cudaMallocManaged(unsigned char** devPtr, size_t size,
										  unsigned int flags = cudaMemAttachGlobal) const {
		return ::cudaMallocManaged(devPtr, size, flags);
	}

	virtual cudaError_t cudaFree(void* devPtr) const { return ::cudaFree(devPtr); }
};
