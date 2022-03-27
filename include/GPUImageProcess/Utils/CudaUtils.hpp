#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class CudaUtils
{
public:
	template <typename T>
	static void freeUnifiedPtr(T* ptr)
	{
		cudaFree(ptr);
	}
};
