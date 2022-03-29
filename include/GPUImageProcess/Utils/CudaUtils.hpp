#pragma once
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdexcept>
#include <string>
#include <map>
#include <sstream>

#define cudaErrorChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		const std::string msg = "ERROR (CUDA) " + std::to_string(line) + ": " + std::string(cudaGetErrorString(code))+ "\n";
		throw std::runtime_error(msg);
	}
}

namespace CudaUtils
{
	template <typename T>
	inline static void freeUnifiedPtr(T* ptr)
	{
		cudaFree(ptr);
	};

	[[nodiscard]] bool isCudaDriverAvailable();
	[[nodiscard]] bool isCudaDeviceAvailable();
	[[nodiscard]] std::map<std::string, std::string> getCudaDeviceInfo();
	[[nodiscard]] std::string getReadableCudaDeviceInfo();
}