#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdexcept>
#include <string>

#define cudaErrorChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		const std::string msg = "ERROR (CUDA) " + std::to_string(line) + ": " + std::string(cudaGetErrorString(code))+ "\n";
		throw std::runtime_error(msg);
	}
}

class CudaUtils
{
public:
	template <typename T>
	static void freeUnifiedPtr(T* ptr)
	{
		cudaFree(ptr);
	}
};
