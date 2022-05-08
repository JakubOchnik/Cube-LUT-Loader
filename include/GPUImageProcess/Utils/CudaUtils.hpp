#pragma once
#ifdef _WIN32
// Workaround for WinAPI colliding with some
// C++17 symbols (caused by using namespace std):
// #define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdexcept>
#include <string>
#include <map>

#define cudaErrorChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(const cudaError_t code, const char *file, const int line)
{
	if (code != cudaSuccess)
	{
		const std::string msg = "ERROR (CUDA) " + std::to_string(code) + ": " + std::string(cudaGetErrorString(code)) + \
			 					"\n";
		throw std::runtime_error(msg);
	}
}

namespace CudaUtils
{
	template <typename T>
	static void freeUnifiedPtr(T* ptr)
	{
		cudaErrorChk(cudaFree(ptr));
	};

	[[nodiscard]] bool isCudaDriverAvailable();
	[[nodiscard]] bool isCudaDeviceAvailable();
	[[nodiscard]] std::map<std::string, std::string> getCudaDeviceInfo();
	[[nodiscard]] std::string getReadableCudaDeviceInfo();
}
