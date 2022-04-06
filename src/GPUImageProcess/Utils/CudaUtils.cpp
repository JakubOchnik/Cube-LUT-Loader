#include <GPUImageProcess/Utils/CudaUtils.hpp>

bool CudaUtils::isCudaDriverAvailable()
{
#ifdef _WIN32
	if (LoadLibraryA("nvcuda.dll") == nullptr)
	{
		return false;
	}
#else
	if (dlopen("libcuda.so", RTLD_NOW) == nullptr) {
		return false;
	}
#endif
	return true;
}

bool CudaUtils::isCudaDeviceAvailable()
{
    cudaDeviceProp prop;
	int devCount{};
	cudaErrorChk(cudaGetDeviceCount(&devCount));
	if (devCount < 1 || cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
	{
		return false;
	}
	return true;
}

std::map<std::string, std::string> CudaUtils::getCudaDeviceInfo()
{
    std::map<std::string, std::string> deviceInfo;
    cudaDeviceProp prop;
    int devCount{};
    cudaErrorChk(cudaGetDeviceCount(&devCount));
    if (devCount < 1 || cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
    {
        return deviceInfo;
    }
    deviceInfo["Device Name"] = prop.name;
    deviceInfo["ComputeCapability"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
    deviceInfo["TotalGlobalMem"] = std::to_string(prop.totalGlobalMem);
    deviceInfo["MaxThreadsPerBlock"] = std::to_string(prop.maxThreadsPerBlock);
    deviceInfo["MaxThreadsDim"] = std::to_string(prop.maxThreadsDim[0]) + "x" + std::to_string(prop.maxThreadsDim[1]) + \
                                "x" + std::to_string(prop.maxThreadsDim[2]);
    deviceInfo["MaxGridSize"] = std::to_string(prop.maxGridSize[0]) + "x" + std::to_string(prop.maxGridSize[1]) + "x" + \
                                std::to_string(prop.maxGridSize[2]);
    return deviceInfo;
}

std::string CudaUtils::getReadableCudaDeviceInfo()
{
    std::stringstream ss;
    auto deviceInfo = CudaUtils::getCudaDeviceInfo();
    for (auto& [key, value] : deviceInfo)
    {
        ss << key << ": " << value << "\n";
    }
    return ss.str();
}
